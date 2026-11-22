//! These tests drive real Monty workers. They need the worker binary in
//! `CODEX_MONTY_BINARY` and are skipped without it.

use std::sync::Arc;
use std::sync::Mutex;

use codex_code_mode_protocol::CellId;
use codex_code_mode_protocol::CodeModeNestedToolCall;
use codex_code_mode_protocol::CodeModeSession;
use codex_code_mode_protocol::CodeModeSessionDelegate;
use codex_code_mode_protocol::CodeModeSessionProvider;
use codex_code_mode_protocol::CodeModeToolKind;
use codex_code_mode_protocol::ExecuteRequest;
use codex_code_mode_protocol::FunctionCallOutputContentItem;
use codex_code_mode_protocol::NotificationFuture;
use codex_code_mode_protocol::RuntimeResponse;
use codex_code_mode_protocol::TerminateRequest;
use codex_code_mode_protocol::ToolDefinition;
use codex_code_mode_protocol::ToolInvocationFuture;
use codex_code_mode_protocol::WaitOutcome;
use codex_code_mode_protocol::WaitRequest;
use codex_protocol::ToolName;
use pretty_assertions::assert_eq;
use serde_json::Value as JsonValue;
use serde_json::json;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use super::MONTY_BINARY_ENV;
use super::MontyCodeModeSessionProvider;

/// Answers `echo` with its input and `gated` once the test releases a permit.
#[derive(Default)]
struct RecordingDelegate {
    calls: Mutex<Vec<(String, Option<JsonValue>)>>,
    notifications: Mutex<Vec<String>>,
    gate: Option<Arc<Semaphore>>,
}

impl CodeModeSessionDelegate for RecordingDelegate {
    fn invoke_tool<'a>(
        &'a self,
        invocation: CodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> ToolInvocationFuture<'a> {
        Box::pin(async move {
            self.calls
                .lock()
                .unwrap()
                .push((invocation.tool_name.name.clone(), invocation.input.clone()));
            match invocation.tool_name.name.as_str() {
                "echo" => Ok(json!({"echo": invocation.input})),
                "fail" => Err("tool exploded".to_string()),
                "gated" => {
                    let gate = self.gate.clone().expect("gated tool needs a gate");
                    tokio::select! {
                        permit = gate.acquire() => {
                            permit.expect("gate closed").forget();
                            Ok(json!("released"))
                        }
                        () = cancellation_token.cancelled() => Err("cancelled".to_string()),
                    }
                }
                other => Err(format!("unknown tool {other}")),
            }
        })
    }

    fn notify<'a>(
        &'a self,
        _call_id: String,
        _cell_id: CellId,
        text: String,
        _cancellation_token: CancellationToken,
    ) -> NotificationFuture<'a> {
        Box::pin(async move {
            self.notifications.lock().unwrap().push(text);
            Ok(())
        })
    }

    fn cell_closed(&self, _cell_id: &CellId) {}
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        tool_name: ToolName::plain(name),
        description: format!("The {name} tool."),
        kind: CodeModeToolKind::Function,
        input_schema: None,
        input_schema_max_bytes: None,
        output_schema: None,
    }
}

fn request(source: &str, yield_time_ms: u64) -> ExecuteRequest {
    ExecuteRequest {
        tool_call_id: "call-1".to_string(),
        enabled_tools: vec![tool("echo"), tool("fail"), tool("gated")],
        source: source.to_string(),
        yield_time_ms: Some(yield_time_ms),
        max_output_tokens: None,
    }
}

async fn session() -> Option<Arc<dyn CodeModeSession>> {
    let Some(binary) = std::env::var_os(MONTY_BINARY_ENV) else {
        eprintln!("skipping: {MONTY_BINARY_ENV} is not set");
        return None;
    };
    let provider = MontyCodeModeSessionProvider::new(Some(binary.into()));
    Some(provider.create_session().await.expect("session"))
}

async fn run(
    session: &Arc<dyn CodeModeSession>,
    delegate: Arc<RecordingDelegate>,
    source: &str,
) -> RuntimeResponse {
    session
        .execute(request(source, 30_000), delegate, None)
        .await
        .expect("execute")
        .initial_response()
        .await
        .expect("response")
}

fn texts(items: &[FunctionCallOutputContentItem]) -> Vec<&str> {
    items
        .iter()
        .map(|item| match item {
            FunctionCallOutputContentItem::InputText { text } => text.as_str(),
            other => panic!("unexpected item {other:?}"),
        })
        .collect()
}

fn completed(response: &RuntimeResponse) -> (Vec<&str>, Option<&str>) {
    match response {
        RuntimeResponse::Result {
            content_items,
            error_text,
            ..
        } => (texts(content_items), error_text.as_deref()),
        other => panic!("expected a result, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn prints_final_expression_and_stores_between_cells() {
    let Some(session) = session().await else {
        return;
    };
    let delegate = Arc::new(RecordingDelegate::default());

    let first = run(
        &session,
        Arc::clone(&delegate),
        "print('hello')\nstore('count', {'n': 41})\ntext(42)\n[1, 2]",
    )
    .await;
    assert_eq!(completed(&first), (vec!["hello\n", "42", "[1, 2]\n"], None));

    let second = run(
        &session,
        Arc::clone(&delegate),
        "value = load('count')\nprint(value['n'] + 1, load('missing'))",
    )
    .await;
    assert_eq!(completed(&second), (vec!["42 None\n"], None));
}

#[tokio::test(flavor = "multi_thread")]
async fn calls_tools_concurrently_and_raises_tool_errors() {
    let Some(session) = session().await else {
        return;
    };
    let delegate = Arc::new(RecordingDelegate::default());

    let response = run(
        &session,
        Arc::clone(&delegate),
        r#"
import asyncio
a, b = await asyncio.gather(tools.echo(x=1), tools.echo({"y": [True, None]}))
print(a["echo"], b["echo"])
try:
    await tools.fail()
except RuntimeError as exc:
    print("caught", exc)
"#,
    )
    .await;
    assert_eq!(
        completed(&response),
        (
            vec!["{'x': 1} {'y': [True, None]}\ncaught tool exploded\n"],
            None
        )
    );
    assert_eq!(
        delegate.calls.lock().unwrap().clone(),
        vec![
            ("echo".to_string(), Some(json!({"x": 1}))),
            ("echo".to_string(), Some(json!({"y": [true, null]}))),
            ("fail".to_string(), Some(json!({}))),
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn reports_exceptions_and_clean_exits() {
    let Some(session) = session().await else {
        return;
    };
    let delegate = Arc::new(RecordingDelegate::default());

    let failed = run(&session, Arc::clone(&delegate), "print('before')\n1 / 0").await;
    let (output, error) = completed(&failed);
    assert_eq!(output, vec!["before\n"]);
    let error = error.expect("error text");
    assert!(error.contains("ZeroDivisionError"), "{error}");

    let exited = run(
        &session,
        Arc::clone(&delegate),
        "print('bye')\nraise SystemExit\nprint('unreachable')",
    )
    .await;
    assert_eq!(completed(&exited), (vec!["bye\n"], None));

    let tools = run(
        &session,
        delegate,
        "print(sorted(t['name'] for t in ALL_TOOLS))",
    )
    .await;
    assert_eq!(
        completed(&tools),
        (vec!["['echo', 'fail', 'gated']\n"], None)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn yields_then_resumes_with_wait() {
    let Some(session) = session().await else {
        return;
    };
    let gate = Arc::new(Semaphore::new(0));
    let delegate = Arc::new(RecordingDelegate {
        gate: Some(Arc::clone(&gate)),
        ..RecordingDelegate::default()
    });

    let started = session
        .execute(
            request(
                "print('waiting')\nyield_control()\nprint(await tools.gated())",
                30_000,
            ),
            delegate,
            None,
        )
        .await
        .expect("execute");
    let cell_id = started.cell_id.clone();
    match started.initial_response().await.expect("response") {
        RuntimeResponse::Yielded { content_items, .. } => {
            assert_eq!(texts(&content_items), vec!["waiting\n"]);
        }
        other => panic!("expected a yield, got {other:?}"),
    }

    gate.add_permits(1);
    let outcome = session
        .wait(
            WaitRequest {
                cell_id: cell_id.clone(),
                yield_time_ms: 30_000,
                max_output_tokens: None,
            },
            /*preempt*/ None,
        )
        .await
        .expect("wait");
    match outcome {
        WaitOutcome::LiveCell(response) => {
            assert_eq!(completed(&response), (vec!["released\n"], None));
        }
        other => panic!("expected a live cell, got {other:?}"),
    }

    let missing = session
        .wait(
            WaitRequest {
                cell_id,
                yield_time_ms: 10,
                max_output_tokens: None,
            },
            /*preempt*/ None,
        )
        .await
        .expect("wait");
    assert!(matches!(missing, WaitOutcome::MissingCell(_)));
}

#[tokio::test(flavor = "multi_thread")]
async fn terminates_a_busy_cell() {
    let Some(session) = session().await else {
        return;
    };
    let delegate = Arc::new(RecordingDelegate::default());

    let started = session
        .execute(
            request("print('spinning')\nwhile True:\n    pass", 200),
            delegate,
            None,
        )
        .await
        .expect("execute");
    let cell_id = started.cell_id.clone();
    assert!(matches!(
        started.initial_response().await.expect("response"),
        RuntimeResponse::Yielded { .. }
    ));

    let outcome = session
        .terminate(TerminateRequest::from(cell_id))
        .await
        .expect("terminate");
    assert!(
        matches!(
            outcome,
            WaitOutcome::LiveCell(RuntimeResponse::Terminated { .. })
        ),
        "{outcome:?}"
    );
}
