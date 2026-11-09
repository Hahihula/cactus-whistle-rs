//! The training dashboard (ratatui). It reads the runs' event files, so it
//! works the same next to the runner (`run`) and on its own (`watch`).

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, Paragraph, Row, Table,
    TableState, Tabs, Wrap,
};
use ratatui::{DefaultTerminal, Frame};

use crate::hints::{Level, hints};
use crate::runner::Control;
use crate::state::{Gpu, RunState, Status, SweepState, TestState, now_secs};

const VIEWS: [&str; 5] = ["Overview", "Curves", "Samples", "Hints", "Log"];

struct App {
    sweep: SweepState,
    gpu: Arc<Mutex<Option<Gpu>>>,
    ctl: Option<Arc<Control>>,
    table: TableState,
    view: usize,
    /// Checkpoint shown in Samples, counted from the latest (0 = latest).
    sample_back: usize,
    log_scroll: u16,
    confirm: Option<char>,
    follow: bool,
}

pub fn gpu_poller(gpu: Arc<Mutex<Option<Gpu>>>) {
    std::thread::spawn(move || {
        loop {
            let g = std::process::Command::new("nvidia-smi")
                .args([
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| {
                    let s = String::from_utf8_lossy(&o.stdout);
                    let v: Vec<f64> = s
                        .lines()
                        .next()?
                        .split(',')
                        .filter_map(|x| x.trim().parse().ok())
                        .collect();
                    (v.len() == 4).then(|| Gpu {
                        util: v[0],
                        mem_used: v[1],
                        mem_total: v[2],
                        temp: v[3],
                    })
                });
            if let Ok(mut m) = gpu.lock() {
                *m = g;
            }
            std::thread::sleep(Duration::from_secs(3));
        }
    });
}

pub fn run(sweep: SweepState, ctl: Option<Arc<Control>>) -> std::io::Result<()> {
    let gpu = Arc::new(Mutex::new(None));
    gpu_poller(gpu.clone());
    let mut app = App {
        sweep,
        gpu,
        ctl,
        table: TableState::default().with_selected(Some(0)),
        view: 0,
        sample_back: 0,
        log_scroll: 0,
        confirm: None,
        follow: true,
    };
    let mut term = ratatui::init();
    let r = app.main_loop(&mut term);
    ratatui::restore();
    r
}

fn fmt_dur(s: f64) -> String {
    let s = s.max(0.0) as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, s % 3600 / 60)
    } else {
        format!("{}m{:02}s", s / 60, s % 60)
    }
}

const SPIN: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

fn spinner() -> &'static str {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    SPIN[(ms / 100 % SPIN.len() as u128) as usize]
}

/// What the evaluator is doing for `r`, in words, if anything.
fn eval_activity(r: &RunState) -> Option<String> {
    if !r.evaluating() {
        return None;
    }
    let now = now_secs();
    let total = r.evals.len();
    let done = r.evaluated.len().min(total);
    match (&r.busy, r.test_state) {
        (Some((w, _, since, clips)), _) if w == "test" => Some(format!(
            "scoring the best adapter on the test split ({clips} clips), running {}",
            fmt_dur(now - since)
        )),
        (Some((_, step, _, _)), t) => Some(format!(
            "evaluating checkpoint {}/{total} (step {step}){}{}",
            done + 1,
            r.eval_eta_secs(now)
                .map_or(String::new(), |e| format!(", ~{} left", fmt_dur(e))),
            if t == TestState::Queued {
                ", then the test split"
            } else {
                ""
            }
        )),
        (None, TestState::Queued) if r.checkpoints_pending() == 0 => {
            Some("test split queued (waiting for the evaluator)".into())
        }
        _ => Some(format!(
            "{} checkpoint(s) waiting for the evaluator{}",
            r.checkpoints_pending(),
            r.eval_eta_secs(now)
                .map_or(String::new(), |e| format!(", ~{}", fmt_dur(e)))
        )),
    }
}

fn status_style(s: Status) -> Style {
    match s {
        Status::Running => Style::new().fg(Color::Cyan).bold(),
        Status::Done => Style::new().fg(Color::Green),
        Status::Failed | Status::Stalled => Style::new().fg(Color::Red).bold(),
        Status::Stopped => Style::new().fg(Color::Yellow),
        Status::Queued => Style::new().fg(Color::DarkGray),
    }
}

impl App {
    fn main_loop(&mut self, term: &mut DefaultTerminal) -> std::io::Result<()> {
        let mut last_poll = Instant::now() - Duration::from_secs(10);
        loop {
            if last_poll.elapsed() > Duration::from_millis(500) {
                self.sweep.poll();
                self.sweep.gpu = self.gpu.lock().ok().and_then(|g| g.clone());
                if let Some(c) = &self.ctl {
                    self.sweep.runner_msg = c.msg.lock().map(|m| m.clone()).unwrap_or_default();
                }
                if self.follow
                    && let Some(i) = self
                        .sweep
                        .runs
                        .iter()
                        .position(|r| r.status == Status::Running)
                {
                    self.table.select(Some(i));
                }
                last_poll = Instant::now();
            }
            term.draw(|f| self.draw(f))?;
            if event::poll(Duration::from_millis(250))?
                && let Event::Key(k) = event::read()?
                && k.kind == KeyEventKind::Press
                && self.key(k.code, k.modifiers)
            {
                return Ok(());
            }
        }
    }

    /// Returns true to quit.
    fn key(&mut self, code: KeyCode, mods: KeyModifiers) -> bool {
        let running = self
            .ctl
            .as_ref()
            .is_some_and(|c| !c.finished.load(Ordering::SeqCst));
        if let Some(c) = self.confirm.take() {
            match (c, code) {
                ('q', KeyCode::Char('q')) | ('q', KeyCode::Char('y')) => {
                    if let Some(ctl) = &self.ctl {
                        ctl.request_stop();
                    }
                    return true;
                }
                ('s', KeyCode::Char('s')) | ('s', KeyCode::Char('y')) => {
                    if let Some(ctl) = &self.ctl {
                        ctl.request_stop();
                    }
                }
                _ => {}
            }
            return false;
        }
        let n = self.sweep.runs.len();
        match code {
            KeyCode::Char('c') if mods.contains(KeyModifiers::CONTROL) => {
                if running {
                    self.confirm = Some('q');
                } else {
                    return true;
                }
            }
            KeyCode::Char('q') | KeyCode::Esc => {
                if running {
                    self.confirm = Some('q');
                } else {
                    return true;
                }
            }
            KeyCode::Char('s') if running => self.confirm = Some('s'),
            KeyCode::Down | KeyCode::Char('j') if n > 0 => {
                self.follow = false;
                let i = self.table.selected().unwrap_or(0);
                self.table.select(Some((i + 1).min(n - 1)));
                self.sample_back = 0;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.follow = false;
                let i = self.table.selected().unwrap_or(0);
                self.table.select(Some(i.saturating_sub(1)));
                self.sample_back = 0;
            }
            KeyCode::Char('f') => self.follow = true,
            KeyCode::Tab => self.view = (self.view + 1) % VIEWS.len(),
            KeyCode::BackTab => self.view = (self.view + VIEWS.len() - 1) % VIEWS.len(),
            KeyCode::Char(c @ '1'..='5') => self.view = c as usize - '1' as usize,
            KeyCode::Left | KeyCode::Char('h') => self.sample_back += 1,
            KeyCode::Right | KeyCode::Char('l') => {
                self.sample_back = self.sample_back.saturating_sub(1)
            }
            KeyCode::PageUp => self.log_scroll = self.log_scroll.saturating_add(10),
            KeyCode::PageDown => self.log_scroll = self.log_scroll.saturating_sub(10),
            _ => {}
        }
        false
    }

    fn selected(&self) -> Option<&RunState> {
        self.table.selected().and_then(|i| self.sweep.runs.get(i))
    }

    fn draw(&mut self, f: &mut Frame) {
        let runs_h = (self.sweep.runs.len() as u16 + 3).min(12);
        let [head, runs, tabs, body, foot] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(runs_h),
            Constraint::Length(1),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .areas(f.area());
        self.draw_header(f, head);
        self.draw_runs(f, runs);
        f.render_widget(
            Tabs::new(
                VIEWS
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("{} {v}", i + 1)),
            )
            .select(self.view)
            .highlight_style(Style::new().fg(Color::Black).bg(Color::Cyan)),
            tabs,
        );
        if let Some(r) = self.selected().cloned() {
            match self.view {
                0 => self.draw_overview(f, body, &r),
                1 => draw_curves(f, body, &r),
                2 => self.draw_samples(f, body, &r),
                3 => self.draw_hints(f, body, &r),
                _ => self.draw_log(f, body, &r),
            }
        } else {
            f.render_widget(Paragraph::new("No runs yet."), body);
        }
        let keys = match self.confirm {
            Some('q') => " Quit? Training stops with its state saved (run again to resume). q/y = yes, any other key = no ".to_string(),
            Some('s') => " Stop the sweep? The trainer saves its state. s/y = yes, any other key = no ".to_string(),
            _ => {
                let mut k = " ↑↓ run  1-5/Tab view  ←→ checkpoint (Samples)  PgUp/PgDn log  f follow".to_string();
                if self.ctl.is_some() {
                    k += "  s stop";
                }
                k + "  q quit "
            }
        };
        let style = if self.confirm.is_some() {
            Style::new().fg(Color::Black).bg(Color::Yellow)
        } else {
            Style::new().fg(Color::DarkGray)
        };
        f.render_widget(Paragraph::new(keys).style(style), foot);
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let mut spans = vec![
            Span::styled(
                " whistle-train ",
                Style::new().fg(Color::Black).bg(Color::Cyan).bold(),
            ),
            Span::raw(format!(" {} ", self.sweep.dir.display())),
        ];
        if !self.sweep.runner_msg.is_empty() {
            spans.push(Span::styled(
                format!(" {} ", self.sweep.runner_msg),
                Style::new().fg(Color::Yellow),
            ));
        } else if self.ctl.is_none() {
            spans.push(Span::styled(" watching ", Style::new().fg(Color::DarkGray)));
        }
        if let Some(g) = &self.sweep.gpu {
            spans.push(Span::raw(format!(
                "  GPU {:.0}%  {:.1}/{:.1} GB  {:.0}°C",
                g.util,
                g.mem_used / 1024.0,
                g.mem_total / 1024.0,
                g.temp
            )));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn draw_runs(&mut self, f: &mut Frame, area: Rect) {
        let header = Row::new([
            "run", "status", "progress", "train", "best val", "WER", "test", "ETA",
        ])
        .style(Style::new().bold().fg(Color::Gray));
        let rows: Vec<Row> = self
            .sweep
            .runs
            .iter()
            .map(|r| {
                let total = r.total_steps();
                let evaluating = r.status == Status::Done && r.evaluating();
                let progress = if evaluating {
                    if r.checkpoints_pending() > 0 {
                        format!(
                            "eval {}/{}",
                            r.evaluated.len().min(r.evals.len()),
                            r.evals.len()
                        )
                    } else {
                        "test split".into()
                    }
                } else if total > 0.0 {
                    format!(
                        "{:.0}%  ep {:.2}",
                        100.0 * r.step() / total,
                        r.steps.last().map_or(0.0, |s| s.epoch)
                    )
                } else {
                    String::new()
                };
                let train = r
                    .steps
                    .last()
                    .map_or(String::new(), |s| format!("{:.4}", s.loss));
                let best = r
                    .best_eval()
                    .map_or(String::new(), |e| format!("{:.4} @{}", e.val_loss, e.step));
                let wer = r
                    .wers
                    .last()
                    .map_or(String::new(), |w| format!("{:.1}%", 100.0 * w.wer));
                let test = r
                    .test
                    .as_ref()
                    .map_or(String::new(), |w| format!("{:.1}%", 100.0 * w.wer));
                let eta = r.eta_secs().map_or_else(
                    || {
                        r.eval_eta_secs(now_secs())
                            .map_or(String::new(), |e| format!("~{}", fmt_dur(e)))
                    },
                    fmt_dur,
                );
                let (status, style) = if evaluating {
                    (
                        format!("eval {}", spinner()),
                        Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                    )
                } else {
                    (r.status.name().to_string(), status_style(r.status))
                };
                let warn = hints(r, None).iter().any(|h| h.level >= Level::Warn);
                Row::new([
                    Cell::from(format!("{}{}", if warn { "! " } else { "  " }, r.name)),
                    Cell::from(status).style(style),
                    Cell::from(progress),
                    Cell::from(train),
                    Cell::from(best),
                    Cell::from(wer),
                    Cell::from(test),
                    Cell::from(eta),
                ])
            })
            .collect();
        let name_w = self
            .sweep
            .runs
            .iter()
            .map(|r| r.name.chars().count())
            .max()
            .unwrap_or(4)
            .max(4) as u16
            + 3;
        let t = Table::new(
            rows,
            [
                Constraint::Length(name_w),
                Constraint::Length(8),
                Constraint::Length(14),
                Constraint::Length(8),
                Constraint::Length(16),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Length(8),
                Constraint::Min(0),
            ],
        )
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(" runs "))
        .row_highlight_style(
            Style::new()
                .bg(Color::Rgb(40, 40, 60))
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("›");
        f.render_stateful_widget(t, area, &mut self.table);
    }

    fn draw_overview(&self, f: &mut Frame, area: Rect, r: &RunState) {
        let [top, rest] = Layout::vertical([Constraint::Length(3), Constraint::Min(4)]).areas(area);
        let total = r.total_steps();
        let ratio = if total > 0.0 {
            (r.step() / total).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut label = format!(
            "step {}/{}  epoch {:.2}  {}",
            r.step(),
            total,
            r.steps.last().map_or(0.0, |s| s.epoch),
            r.eta_secs()
                .map_or(String::new(), |e| format!("ETA {}", fmt_dur(e)))
        );
        let mut title = format!(" {} — {} ", r.name, r.status.name());
        let (mut ratio, mut color) = (ratio, Color::Cyan);
        if r.status == Status::Done
            && let Some(what) = eval_activity(r)
        {
            // Training is over but results are still coming: say so, and move.
            title = format!(" {} — training done, evaluation running ", r.name);
            label = format!("{} {what}", spinner());
            color = Color::Yellow;
            ratio = if r.evals.is_empty() {
                1.0
            } else {
                (r.evaluated.len() as f64 / r.evals.len() as f64).clamp(0.0, 1.0)
            };
        }
        f.render_widget(
            Gauge::default()
                .block(Block::default().borders(Borders::ALL).title(title))
                .gauge_style(Style::new().fg(color))
                .ratio(ratio)
                .label(label),
            top,
        );
        let [left, right] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(rest);

        let mut lines: Vec<Line> = Vec::new();
        let kv = |k: &str, v: String| {
            Line::from(vec![
                Span::styled(format!("{k:<16}"), Style::new().fg(Color::Gray)),
                Span::raw(v),
            ])
        };
        if let Some(s) = r.steps.last() {
            lines.push(kv(
                "train loss",
                format!("{:.4}  (max in window {:.3})", s.loss, s.loss_max),
            ));
            lines.push(kv("learning rate", format!("{:.2e}", s.lr)));
            lines.push(kv(
                "grad norm",
                format!("{:.3}  clipped {:.0}%", s.grad_norm, 100.0 * s.clipped),
            ));
        }
        if let Some(e) = r.evals.last() {
            lines.push(kv(
                "val loss",
                format!("{:.4} at step {}", e.val_loss, e.step),
            ));
        }
        if let Some(b) = r.best_eval() {
            lines.push(kv(
                "best val loss",
                format!("{:.4} at step {} (adapter_best)", b.val_loss, b.step),
            ));
        }
        if let Some(w) = r.wers.last() {
            let best = r.wers.iter().map(|x| x.wer).fold(f64::INFINITY, f64::min);
            lines.push(kv(
                "WER (engine)",
                format!(
                    "{:.1}%  strict {:.1}%  best {:.1}%  on {} val clips",
                    100.0 * w.wer,
                    100.0 * w.strict,
                    100.0 * best,
                    w.clips
                ),
            ));
            if !w.by_source.is_empty() {
                let by: Vec<String> = w
                    .by_source
                    .iter()
                    .map(|(k, v)| format!("{k} {:.1}%", 100.0 * v))
                    .collect();
                lines.push(kv("  by source", by.join("  ")));
            }
        }
        if let Some(t) = &r.test {
            lines.push(kv(
                "test WER",
                format!(
                    "{:.1}%  strict {:.1}%  ({} clips)",
                    100.0 * t.wer,
                    100.0 * t.strict,
                    t.clips
                ),
            ));
        }
        if let Some(a) = eval_activity(r) {
            lines.push(kv("evaluator", format!("{} {a}", spinner())));
        }
        if let Some(s) = r.secs_per_step() {
            lines.push(kv("speed", format!("{:.3} s/step", s)));
        }
        if let (Some(a), b) = (r.first_t, r.last_t) {
            lines.push(kv("elapsed", fmt_dur(b - a)));
        }
        if let Some(s) = &r.start {
            lines.push(kv(
                "data",
                format!(
                    "{} train / {} val clips, {:.1} h",
                    s["train_clips"],
                    s["val_clips"],
                    s["minutes"].as_f64().unwrap_or(0.0) / 60.0
                ),
            ));
            lines.push(kv(
                "adapter",
                format!(
                    "{} groups, rank {}, +{} vocabulary rows",
                    s["lora_groups"], s["rank"], s["rows"]
                ),
            ));
        }
        let hs = hints(r, self.sweep.gpu.as_ref());
        if !hs.is_empty() {
            lines.push(Line::raw(""));
            for h in hs.iter().take(3) {
                lines.push(hint_line(h.level, &h.text));
            }
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(" now ")),
            left,
        );

        let mut cfg: Vec<Line> = Vec::new();
        if let Some(m) = r.settings.as_object() {
            for (k, v) in m {
                if matches!(
                    k.as_str(),
                    "train" | "validation" | "out_dir" | "base" | "tokens"
                ) {
                    continue;
                }
                let v = match v {
                    serde_json::Value::Object(o) => o
                        .iter()
                        .map(|(a, b)| format!("{a}={b}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                    other => other.to_string(),
                };
                cfg.push(Line::from(vec![
                    Span::styled(format!("{k:<18}"), Style::new().fg(Color::Gray)),
                    Span::raw(v),
                ]));
            }
        }
        f.render_widget(
            Paragraph::new(cfg).wrap(Wrap { trim: false }).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" settings (others: trainer defaults) "),
            ),
            right,
        );
    }

    fn draw_samples(&self, f: &mut Frame, area: Rect, r: &RunState) {
        if r.samples.is_empty() {
            f.render_widget(
                Paragraph::new("No sample decodes yet: they appear after each checkpoint (add recordings under [eval] samples).")
                    .block(Block::default().borders(Borders::ALL).title(" samples ")),
                area,
            );
            return;
        }
        let i = r.samples.len() - 1 - self.sample_back.min(r.samples.len() - 1);
        let (step, items) = &r.samples[i];
        let prev = i.checked_sub(1).map(|j| &r.samples[j]);
        let mut lines = Vec::new();
        for (k, s) in items.iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(format!("{}  ", s.audio), Style::new().bold()),
                Span::styled(format!("[{}]", s.lang), Style::new().fg(Color::Gray)),
            ]));
            let (against, what) = match (&s.reference, prev.and_then(|(_, p)| p.get(k))) {
                (Some(rf), _) => (Some(rf.clone()), "reference"),
                (None, Some(p)) => (Some(p.hyp.clone()), "previous checkpoint"),
                _ => (None, ""),
            };
            if let Some(rf) = &s.reference {
                lines.push(Line::from(vec![
                    Span::styled("  ref  ", Style::new().fg(Color::Gray)),
                    Span::raw(rf.clone()),
                ]));
            }
            let mut spans = vec![Span::styled("  hyp  ", Style::new().fg(Color::Gray))];
            match &against {
                Some(a) => spans.extend(diff_spans(a, &s.hyp)),
                None => spans.push(Span::raw(s.hyp.clone())),
            }
            lines.push(Line::from(spans));
            if let Some(a) = &against {
                let mut c = cactus_whistle_common::wer::Count::default();
                c.add(a, &s.hyp);
                lines.push(Line::styled(
                    format!(
                        "       {} word changes vs {what} ({:.0}%)",
                        c.errors,
                        100.0 * c.wer()
                    ),
                    Style::new().fg(Color::DarkGray),
                ));
            }
            lines.push(Line::raw(""));
        }
        let title = format!(
            " samples at step {step} ({}/{})  ←→ other checkpoints  colours: changed / added / removed words ",
            i + 1,
            r.samples.len()
        );
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(title)),
            area,
        );
    }

    fn draw_hints(&self, f: &mut Frame, area: Rect, r: &RunState) {
        let hs = hints(r, self.sweep.gpu.as_ref());
        let lines: Vec<Line> = if hs.is_empty() {
            vec![Line::raw(
                "Nothing to remark on. Hints appear here when the numbers suggest something to change.",
            )]
        } else {
            hs.iter()
                .flat_map(|h| [hint_line(h.level, &h.text), Line::raw("")])
                .collect()
        };
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(" hints ")),
            area,
        );
    }

    fn draw_log(&self, f: &mut Frame, area: Rect, r: &RunState) {
        let h = area.height.saturating_sub(2) as usize;
        let n = r.log.len();
        let end = n.saturating_sub(self.log_scroll as usize);
        let lines: Vec<Line> = r
            .log
            .iter()
            .skip(end.saturating_sub(h))
            .take(h.min(end))
            .map(|l| {
                if l.starts_with("ERROR") {
                    Line::styled(l.clone(), Style::new().fg(Color::Red))
                } else if l.starts_with("  ") {
                    Line::styled(l.clone(), Style::new().fg(Color::DarkGray))
                } else {
                    Line::raw(l.clone())
                }
            })
            .collect();
        f.render_widget(
            Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(format!(
                " log ({}/stderr.txt has the trainer's full output) ",
                r.dir.display()
            ))),
            area,
        );
    }
}

fn hint_line(level: Level, text: &str) -> Line<'static> {
    let (tag, style) = match level {
        Level::Bad => ("✖ ", Style::new().fg(Color::Red).bold()),
        Level::Warn => ("▲ ", Style::new().fg(Color::Yellow)),
        Level::Info => ("● ", Style::new().fg(Color::Cyan)),
    };
    Line::from(vec![Span::styled(tag, style), Span::raw(text.to_string())])
}

/// Hypothesis words coloured against `reference` (word alignment by edit
/// distance on normalised words): substitutions yellow, insertions red,
/// deletions as struck-through reference words.
fn diff_spans(reference: &str, hyp: &str) -> Vec<Span<'static>> {
    use cactus_whistle_common::wer::normalize;
    let r: Vec<&str> = reference.split_whitespace().collect();
    let h: Vec<&str> = hyp.split_whitespace().collect();
    let rn: Vec<String> = r.iter().map(|w| normalize(w)).collect();
    let hn: Vec<String> = h.iter().map(|w| normalize(w)).collect();
    let (n, m) = (r.len(), h.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, v) in d[0].iter_mut().enumerate() {
        *v = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = d[i - 1][j - 1] + usize::from(rn[i - 1] != hn[j - 1]);
            d[i][j] = sub.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
        }
    }
    let (mut i, mut j) = (n, m);
    let mut out = Vec::new();
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && d[i][j] == d[i - 1][j - 1] + usize::from(rn[i - 1] != hn[j - 1]) {
            let style = if rn[i - 1] == hn[j - 1] {
                Style::new()
            } else {
                Style::new().fg(Color::Yellow)
            };
            out.push(Span::styled(format!("{} ", h[j - 1]), style));
            i -= 1;
            j -= 1;
        } else if j > 0 && d[i][j] == d[i][j - 1] + 1 {
            out.push(Span::styled(
                format!("{} ", h[j - 1]),
                Style::new().fg(Color::Red),
            ));
            j -= 1;
        } else {
            out.push(Span::styled(
                format!("{} ", r[i - 1]),
                Style::new()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::CROSSED_OUT),
            ));
            i -= 1;
        }
    }
    out.reverse();
    out
}

fn bounds(pts: &[(f64, f64)]) -> ([f64; 2], [f64; 2]) {
    let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for &(x, y) in pts {
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    if pts.is_empty() {
        return ([0.0, 1.0], [0.0, 1.0]);
    }
    let pad = ((y1 - y0) * 0.08).max(1e-3);
    (
        [x0.min(0.0), x1.max(x0 + 1.0)],
        [(y0 - pad).max(0.0), y1 + pad],
    )
}

fn labels(b: [f64; 2], fmt: impl Fn(f64) -> String) -> Vec<Line<'static>> {
    (0..3)
        .map(|i| Line::raw(fmt(b[0] + (b[1] - b[0]) * i as f64 / 2.0)))
        .collect()
}

fn draw_curves(f: &mut Frame, area: Rect, r: &RunState) {
    let [top, bottom] =
        Layout::vertical([Constraint::Percentage(60), Constraint::Percentage(40)]).areas(area);
    // Training loss smoothed over a few events, validation loss per evaluation.
    let raw: Vec<(f64, f64)> = r.steps.iter().map(|p| (p.step, p.loss)).collect();
    let k = (raw.len() / 60).max(1);
    let smooth: Vec<(f64, f64)> = (0..raw.len())
        .map(|i| {
            let lo = i.saturating_sub(k);
            let w = &raw[lo..=i];
            (
                raw[i].0,
                w.iter().map(|p| p.1).sum::<f64>() / w.len() as f64,
            )
        })
        .collect();
    let val: Vec<(f64, f64)> = r.evals.iter().map(|e| (e.step, e.val_loss)).collect();
    let best: Vec<(f64, f64)> = r
        .best_eval()
        .map(|e| vec![(e.step, e.val_loss)])
        .unwrap_or_default();
    let all: Vec<(f64, f64)> = smooth.iter().chain(&val).copied().collect();
    let (xb, yb) = bounds(&all);
    let xb = [0.0, xb[1].max(r.total_steps())];
    let chart = Chart::new(vec![
        Dataset::default()
            .name("train")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::new().fg(Color::Cyan))
            .data(&smooth),
        Dataset::default()
            .name("validation")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::new().fg(Color::Yellow))
            .data(&val),
        Dataset::default()
            .name("best")
            .marker(Marker::Dot)
            .graph_type(GraphType::Scatter)
            .style(Style::new().fg(Color::Green).bold())
            .data(&best),
    ])
    .block(Block::default().borders(Borders::ALL).title(format!(
        " loss  (lr now {}) ",
        r.steps.last().map_or("-".into(), |s| format!("{:.2e}", s.lr))
    )))
    .x_axis(
        Axis::default()
            .bounds(xb)
            .labels(labels(xb, |v| format!("{v:.0}")))
            .title("step"),
    )
    .y_axis(
        Axis::default()
            .bounds(yb)
            .labels(labels(yb, |v| format!("{v:.3}"))),
    );
    f.render_widget(chart, top);

    let w: Vec<(f64, f64)> = r.wers.iter().map(|p| (p.step, 100.0 * p.wer)).collect();
    let ws: Vec<(f64, f64)> = r.wers.iter().map(|p| (p.step, 100.0 * p.strict)).collect();
    let all: Vec<(f64, f64)> = w.iter().chain(&ws).copied().collect();
    let (_, yb) = bounds(&all);
    let chart = Chart::new(vec![
        Dataset::default()
            .name("WER")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::new().fg(Color::Magenta))
            .data(&w),
        Dataset::default()
            .name("strict")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::new().fg(Color::DarkGray))
            .data(&ws),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" WER on validation clips, Rust engine + runtime LoRA (%) "),
    )
    .x_axis(
        Axis::default()
            .bounds(xb)
            .labels(labels(xb, |v| format!("{v:.0}"))),
    )
    .y_axis(
        Axis::default()
            .bounds(yb)
            .labels(labels(yb, |v| format!("{v:.1}"))),
    );
    f.render_widget(chart, bottom);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    /// Renders every view of the sweep in $WHISTLE_TRAIN_RUNS as text
    /// (`cargo test -p whistle-train render -- --nocapture`); skipped without it.
    #[test]
    fn render_views() {
        let Ok(dir) = std::env::var("WHISTLE_TRAIN_RUNS") else {
            return;
        };
        let mut sweep = SweepState::open(std::path::Path::new(&dir));
        sweep.poll();
        let mut app = App {
            sweep,
            gpu: Arc::new(Mutex::new(None)),
            ctl: None,
            table: TableState::default().with_selected(Some(0)),
            view: 0,
            sample_back: 0,
            log_scroll: 0,
            confirm: None,
            follow: false,
        };
        for view in 0..VIEWS.len() {
            app.view = view;
            let mut t = Terminal::new(TestBackend::new(150, 44)).unwrap();
            t.draw(|f| app.draw(f)).unwrap();
            let buf = t.backend().buffer();
            println!("===== view {view}");
            for y in 0..buf.area.height {
                let line: String = (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect();
                println!("{}", line.trim_end());
            }
        }
    }
}
