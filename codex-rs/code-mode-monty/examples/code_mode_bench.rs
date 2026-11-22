//! Compares Code Mode latency between the JavaScript host (V8, gRPC over stdio, as Core
//! runs it) and Monty (pooled Python workers).
//!
//! ```sh
//! cargo run --release -p codex-code-mode-monty --example code_mode_bench -- \
//!     --js-host /path/to/codex-code-mode-host --monty /path/to/monty
//! ```

// A benchmark: failing loudly is the right response to any surprise.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::io;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use codex_code_mode::CellId;
use codex_code_mode::CodeModeNestedToolCall;
use codex_code_mode::CodeModeSession;
use codex_code_mode::CodeModeSessionDelegate;
use codex_code_mode::CodeModeSessionProvider;
use codex_code_mode::CodeModeToolKind;
use codex_code_mode::ExecuteRequest;
use codex_code_mode::FunctionCallOutputContentItem;
use codex_code_mode::GrpcCodeModeSessionProvider;
use codex_code_mode::NotificationFuture;
use codex_code_mode::RuntimeResponse;
use codex_code_mode::ToolDefinition;
use codex_code_mode::ToolInvocationFuture;
use codex_code_mode_monty::MontyCodeModeSessionProvider;
use codex_protocol::ToolName;
use hyper_util::rt::TokioIo;
use tokio::process::Child;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;
use tonic::transport::Endpoint;
use tower::service_fn;

struct EchoDelegate;

impl CodeModeSessionDelegate for EchoDelegate {
    fn invoke_tool<'a>(
        &'a self,
        invocation: CodeModeNestedToolCall,
        _cancellation_token: CancellationToken,
    ) -> ToolInvocationFuture<'a> {
        Box::pin(async move { Ok(invocation.input.unwrap_or_default()) })
    }

    fn notify<'a>(
        &'a self,
        _call_id: String,
        _cell_id: CellId,
        _text: String,
        _cancellation_token: CancellationToken,
    ) -> NotificationFuture<'a> {
        Box::pin(async { Ok(()) })
    }

    fn cell_closed(&self, _cell_id: &CellId) {}
}

#[derive(Clone, Copy, PartialEq)]
enum Backend {
    JavaScript,
    Python,
}

impl Backend {
    fn label(self) -> &'static str {
        match self {
            Self::JavaScript => "JS (V8 host)",
            Self::Python => "Python (Monty)",
        }
    }
}

struct Paths {
    js_host: PathBuf,
    monty: PathBuf,
}

/// Keeps the spawned JS host alive for as long as its provider is in use.
struct Provider {
    inner: Arc<dyn CodeModeSessionProvider>,
    _child: Option<Child>,
}

fn provider(backend: Backend, paths: &Paths) -> io::Result<Provider> {
    match backend {
        Backend::Python => Ok(Provider {
            inner: Arc::new(MontyCodeModeSessionProvider::new(Some(paths.monty.clone()))),
            _child: None,
        }),
        Backend::JavaScript => {
            // Same launch as Core: one child per channel, gRPC over its stdio.
            let mut child = Command::new(&paths.js_host)
                .args(["--listen", "grpc+stdio://"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .kill_on_drop(true)
                .spawn()?;
            let stdin = child.stdin.take().expect("stdin");
            let stdout = child.stdout.take().expect("stdout");
            let io = Arc::new(Mutex::new(Some(tokio::io::join(stdout, stdin))));
            let channel = Endpoint::from_static("http://code-mode-host")
                .connect_with_connector_lazy(service_fn(move |_| {
                    let io = io.lock().unwrap().take();
                    async move {
                        io.map(TokioIo::new).ok_or_else(|| {
                            io::Error::new(io::ErrorKind::NotConnected, "host connection used")
                        })
                    }
                }));
            Ok(Provider {
                inner: Arc::new(GrpcCodeModeSessionProvider::with_channel(channel)),
                _child: Some(child),
            })
        }
    }
}

fn echo_tool() -> ToolDefinition {
    ToolDefinition {
        name: "echo".to_string(),
        tool_name: ToolName::plain("echo"),
        description: "Returns its input.".to_string(),
        kind: CodeModeToolKind::Function,
        input_schema: None,
        input_schema_max_bytes: None,
        output_schema: None,
    }
}

async fn exec(session: &Arc<dyn CodeModeSession>, source: &str) -> String {
    let started = session
        .execute(
            ExecuteRequest {
                tool_call_id: "bench".to_string(),
                enabled_tools: vec![echo_tool()],
                source: source.to_string(),
                yield_time_ms: Some(120_000),
                max_output_tokens: None,
            },
            Arc::new(EchoDelegate),
            None,
        )
        .await
        .expect("execute");
    match started.initial_response().await.expect("response") {
        RuntimeResponse::Result {
            content_items,
            error_text,
            ..
        } => {
            if let Some(error) = error_text {
                panic!("cell failed: {error}\n{source}");
            }
            content_items
                .iter()
                .filter_map(|item| match item {
                    FunctionCallOutputContentItem::InputText { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        }
        other => panic!("cell did not complete: {other:?}"),
    }
}

struct Stats {
    median: Duration,
    p95: Duration,
    min: Duration,
}

fn stats(mut samples: Vec<Duration>) -> Stats {
    samples.sort();
    let pick = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    Stats {
        median: pick(0.5),
        p95: pick(0.95),
        min: samples[0],
    }
}

fn fmt(duration: Duration) -> String {
    let micros = duration.as_secs_f64() * 1e6;
    if micros >= 10_000.0 {
        format!("{:.1} ms", micros / 1000.0)
    } else if micros >= 1000.0 {
        format!("{:.2} ms", micros / 1000.0)
    } else {
        format!("{micros:.0} µs")
    }
}

fn row(name: &str, backend: Backend, stats: &Stats) {
    println!(
        "| {name:<34} | {:<14} | {:>9} | {:>9} | {:>9} |",
        backend.label(),
        fmt(stats.median),
        fmt(stats.p95),
        fmt(stats.min)
    );
}

struct Workload {
    name: &'static str,
    js: String,
    python: String,
    iterations: usize,
}

fn workloads() -> Vec<Workload> {
    vec![
        Workload {
            name: "trivial cell (text)",
            js: "text('hi')".into(),
            python: "text('hi')".into(),
            iterations: 300,
        },
        Workload {
            name: "1 tool call",
            js: "text((await tools.echo({i: 1})).i)".into(),
            python: "text((await tools.echo(i=1))['i'])".into(),
            iterations: 300,
        },
        Workload {
            name: "100 sequential tool calls",
            js: "let s = 0; for (let i = 0; i < 100; i++) { s += (await tools.echo({i})).i } text(s)"
                .into(),
            python: "s = 0\nfor i in range(100):\n    s += (await tools.echo(i=i))['i']\ntext(s)"
                .into(),
            iterations: 30,
        },
        Workload {
            name: "100 concurrent tool calls",
            js: "const r = await Promise.all(Array.from({length: 100}, (_, i) => tools.echo({i}))); text(r.reduce((a, x) => a + x.i, 0))"
                .into(),
            python: "import asyncio\nr = await asyncio.gather(*[tools.echo(i=i) for i in range(100)])\ntext(sum(x['i'] for x in r))"
                .into(),
            iterations: 30,
        },
        Workload {
            name: "loop: sum range(1_000_000)",
            js: "let s = 0; for (let i = 0; i < 1000000; i++) { s += i } text(s)".into(),
            python: "s = 0\nfor i in range(1_000_000):\n    s += i\ntext(s)".into(),
            iterations: 10,
        },
        Workload {
            name: "json.dumps 10k dicts",
            js: "text(JSON.stringify(Array.from({length: 10000}, (_, i) => ({i, s: String(i)}))).length)"
                .into(),
            python: "import json\ntext(len(json.dumps([{'i': i, 's': str(i)} for i in range(10000)], separators=(',', ':'))))"
                .into(),
            iterations: 10,
        },
        Workload {
            name: "string building + sort 50k",
            js: "const a = []; for (let i = 0; i < 50000; i++) a.push(`k${(i * 7919) % 50000}`); a.sort(); text(a[123])"
                .into(),
            python: "a = [f'k{(i * 7919) % 50000}' for i in range(50000)]\na.sort()\ntext(a[123])"
                .into(),
            iterations: 10,
        },
    ]
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |flag: &str| {
        args.iter()
            .position(|arg| arg == flag)
            .and_then(|index| args.get(index + 1))
            .map(PathBuf::from)
            .unwrap_or_else(|| panic!("missing {flag} <path>"))
    };
    let paths = Paths {
        js_host: arg("--js-host"),
        monty: arg("--monty"),
    };
    let backends = [Backend::JavaScript, Backend::Python];

    println!(
        "## Cold start (fresh provider each time: spawn host / pool, open session, run `text('hi')`)\n"
    );
    println!(
        "| step                               | backend        |    median |       p95 |       min |"
    );
    println!(
        "|------------------------------------|----------------|-----------|-----------|-----------|"
    );
    for backend in backends {
        let mut sessions = Vec::new();
        let mut first_cells = Vec::new();
        let mut totals = Vec::new();
        for _ in 0..15 {
            let start = Instant::now();
            let provider = provider(backend, &paths).expect("provider");
            let session = provider.inner.create_session().await.expect("session");
            let opened = Instant::now();
            assert_eq!(exec(&session, "text('hi')").await, "hi");
            let done = Instant::now();
            sessions.push(opened - start);
            first_cells.push(done - opened);
            totals.push(done - start);
            let _ = session.shutdown().await;
            drop(session);
            drop(provider);
            // Let killed processes go before the next round.
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        row("start + open session", backend, &stats(sessions));
        row("first cell", backend, &stats(first_cells));
        row("total to first result", backend, &stats(totals));
    }

    println!("\n## Warm (one provider and session, cells run back to back)\n");
    println!(
        "| workload                           | backend        |    median |       p95 |       min |"
    );
    println!(
        "|------------------------------------|----------------|-----------|-----------|-----------|"
    );
    let mut warm = Vec::new();
    for backend in backends {
        let provider = provider(backend, &paths).expect("provider");
        let session = provider.inner.create_session().await.expect("session");
        // Warm-up: the first cells pay one-off costs measured above.
        for _ in 0..20 {
            exec(&session, "text('hi')").await;
        }
        let mut new_sessions = Vec::new();
        for _ in 0..50 {
            let start = Instant::now();
            let extra = provider.inner.create_session().await.expect("session");
            new_sessions.push(start.elapsed());
            let _ = extra.shutdown().await;
        }
        warm.push((backend, "open another session", stats(new_sessions)));
        for workload in workloads() {
            let source = match backend {
                Backend::JavaScript => &workload.js,
                Backend::Python => &workload.python,
            };
            let expected = exec(&session, source).await;
            let mut samples = Vec::with_capacity(workload.iterations);
            for _ in 0..workload.iterations {
                let start = Instant::now();
                let output = exec(&session, source).await;
                samples.push(start.elapsed());
                assert_eq!(output, expected);
            }
            warm.push((backend, workload.name, stats(samples)));
        }
        let _ = session.shutdown().await;
    }
    let mut names: Vec<&str> = Vec::new();
    for (_, name, _) in &warm {
        if !names.contains(name) {
            names.push(name);
        }
    }
    for name in names {
        for (backend, workload, stats) in &warm {
            if *workload == name {
                row(name, *backend, stats);
            }
        }
    }
}
