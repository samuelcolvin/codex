//! One `exec` cell: a Python script on a pooled Monty worker, plus the host side of
//! every tool call and helper it makes.
//!
//! Nested tools are async host functions. A call registers a future in the sandbox and
//! starts the tool on a Tokio task, so `asyncio.gather` runs tools concurrently. When every
//! sandbox task is blocked, Monty asks for results ([`TurnEvent::ResolveFutures`]) and the
//! cell answers with whichever tool calls have finished.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;

use codex_code_mode_protocol::CellId;
use codex_code_mode_protocol::CodeModeNestedToolCall;
use codex_code_mode_protocol::CodeModeSessionDelegate;
use codex_code_mode_protocol::CodeModeToolKind;
use codex_code_mode_protocol::FunctionCallOutputContentItem;
use codex_code_mode_protocol::ImageDetail;
use codex_code_mode_protocol::TOOLS_NAMESPACE;
use codex_code_mode_protocol::ToolDefinition;
use codex_code_mode_protocol::python_identifier;
use monty_pool::Checkout;
use monty_pool::Pool;
use monty_pool::PoolError;
use monty_pool::ReplConfig;
use monty_pool::ResumeValue;
use monty_pool::TurnEvent;
use monty_pool::on_print_sync;
use monty_types::CallArgs;
use monty_types::ExcType;
use monty_types::MontyException;
use monty_types::MontyObject;
use monty_types::MontyUuid;
use monty_types::NameLookupResult;
use monty_types::NamedValues;
use monty_types::ResourceLimits;
use serde_json::Value as JsonValue;
use tokio::sync::Notify;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::convert::display_text;
use crate::convert::json_to_monty;
use crate::convert::monty_to_json;

/// Name the sandbox reports in tracebacks.
const SCRIPT_NAME: &str = "exec.py";
/// Live allocations a cell may hold when the session sets no heap limit.
const DEFAULT_MAX_MEMORY_BYTES: usize = 1024 * 1024 * 1024;
/// Every tool call and helper call is a suspension; scripts that loop over tools make many.
const MAX_SUSPENSIONS: usize = 1_000_000;
/// Fixed id of the `tools` module object; it only has to be unique within one feed.
const TOOLS_MODULE_ID: MontyUuid = MontyUuid::from_u128(0x636f_6465_785f_746f_6f6c_735f_6d6f_6475);

const HELPER_TEXT: &str = "text";
const HELPER_IMAGE: &str = "image";
const HELPER_STORE: &str = "store";
const HELPER_LOAD: &str = "load";
const HELPER_NOTIFY: &str = "notify";
const HELPER_YIELD: &str = "yield_control";

/// What a finished cell reports.
#[derive(Debug)]
pub(crate) enum CellOutcome {
    /// The script ran to the end. `error_text` is set when it raised.
    Completed { error_text: Option<String> },
    /// `terminate` stopped the script.
    Terminated,
}

/// Output and status shared between the running script and whoever observes it.
#[derive(Default)]
pub(crate) struct CellShared {
    state: Mutex<CellState>,
    pub(crate) changed: Notify,
}

#[derive(Default)]
pub(crate) struct CellState {
    items: Vec<FunctionCallOutputContentItem>,
    /// `print()` output not yet turned into an item.
    pending_text: String,
    pub(crate) yield_requested: bool,
    pub(crate) outcome: Option<CellOutcome>,
}

impl CellState {
    /// Takes everything produced since the last observation.
    pub(crate) fn drain_items(&mut self) -> Vec<FunctionCallOutputContentItem> {
        self.flush_text();
        std::mem::take(&mut self.items)
    }

    fn flush_text(&mut self) {
        if !self.pending_text.is_empty() {
            let text = std::mem::take(&mut self.pending_text);
            self.items
                .push(FunctionCallOutputContentItem::InputText { text });
        }
    }
}

impl CellShared {
    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, CellState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn update(&self, change: impl FnOnce(&mut CellState)) {
        change(&mut self.lock());
        self.changed.notify_waiters();
    }

    fn print(&self, text: &str) {
        self.update(|state| state.pending_text.push_str(text));
    }

    fn push_item(&self, item: FunctionCallOutputContentItem) {
        self.update(|state| {
            state.flush_text();
            state.items.push(item);
        });
    }

    pub(crate) fn finish(&self, outcome: CellOutcome) {
        self.update(|state| {
            if state.outcome.is_none() {
                state.outcome = Some(outcome);
            }
        });
    }
}

/// Everything a cell needs to run.
pub(crate) struct CellContext {
    pub(crate) pool: Arc<Pool>,
    pub(crate) cell_id: CellId,
    /// The model's `exec` call, used as the call id of `notify` outputs.
    pub(crate) tool_call_id: String,
    pub(crate) source: String,
    pub(crate) tools: Vec<ToolDefinition>,
    pub(crate) delegate: Arc<dyn CodeModeSessionDelegate>,
    pub(crate) stored: Arc<Mutex<HashMap<String, JsonValue>>>,
    pub(crate) shared: Arc<CellShared>,
    pub(crate) cancel: CancellationToken,
    pub(crate) max_memory_bytes: Option<usize>,
}

/// Runs the script to completion or until the cell is cancelled, then records the outcome.
pub(crate) async fn run_cell(context: CellContext) {
    let shared = Arc::clone(&context.shared);
    let cancel = context.cancel.clone();
    let delegate = Arc::clone(&context.delegate);
    let cell_id = context.cell_id.clone();
    let outcome = tokio::select! {
        // Dropping the script future drops its checkout, which kills the worker, and its
        // join set, which aborts tool calls still in flight.
        () = cancel.cancelled() => CellOutcome::Terminated,
        error_text = run_script(context) => CellOutcome::Completed { error_text },
    };
    // Tool calls observe this token; make sure none outlive the cell.
    cancel.cancel();
    shared.finish(outcome);
    delegate.cell_closed(&cell_id);
}

struct ToolEntry {
    definition: ToolDefinition,
}

struct Runner {
    context: CellContext,
    tools: HashMap<String, ToolEntry>,
    in_flight: JoinSet<(u32, ResumeValue)>,
    finished: HashMap<u32, ResumeValue>,
    next_runtime_call: u64,
}

async fn run_script(context: CellContext) -> Option<String> {
    let mut tools = HashMap::new();
    for definition in &context.tools {
        tools
            .entry(python_identifier(&definition.name))
            .or_insert_with(|| ToolEntry {
                definition: definition.clone(),
            });
    }
    let mut runner = Runner {
        context,
        tools,
        in_flight: JoinSet::new(),
        finished: HashMap::new(),
        next_runtime_call: 0,
    };
    runner.run().await
}

impl Runner {
    async fn run(&mut self) -> Option<String> {
        let limits = ResourceLimits {
            max_memory: Some(
                self.context
                    .max_memory_bytes
                    .unwrap_or(DEFAULT_MAX_MEMORY_BYTES),
            ),
            max_suspensions: MAX_SUSPENSIONS,
            ..ResourceLimits::default()
        };
        let repl = ReplConfig {
            script_name: SCRIPT_NAME.to_string(),
            limits: Some(limits),
            ..ReplConfig::default()
        };
        let mut checkout = match self.context.pool.checkout(&repl).await {
            Ok(checkout) => checkout,
            Err(err) => return Some(format!("failed to start the Python sandbox: {err}")),
        };
        let shared = Arc::clone(&self.context.shared);
        let mut on_print = on_print_sync(move |_stream, text| shared.print(text));

        let inputs = self.inputs();
        let source = std::mem::take(&mut self.context.source);
        let mut event = checkout
            .feed(
                source,
                inputs,
                Vec::new(),
                /*skip_type_check*/ true,
                &mut on_print,
            )
            .await;
        let result = loop {
            event = match event {
                Ok(TurnEvent::Complete(value)) => {
                    if value != MontyObject::none() {
                        self.context.shared.print(&ensure_newline(value.py_repr()));
                    }
                    break None;
                }
                Ok(TurnEvent::FunctionCall {
                    function_name,
                    args,
                    call_id,
                    object_id: None,
                    ..
                }) => {
                    let answer = self.call(&function_name, args, call_id).await;
                    checkout.resume(answer, &mut on_print).await
                }
                Ok(TurnEvent::FunctionCall { .. }) => {
                    checkout.resume(ResumeValue::NotFound, &mut on_print).await
                }
                Ok(TurnEvent::ResolveFutures {
                    pending_call_ids, ..
                }) => match self.settle(&pending_call_ids).await {
                    Ok(results) => checkout.resume_futures(results, &mut on_print).await,
                    Err(message) => break Some(message),
                },
                Ok(TurnEvent::OsCall {
                    system_sleep: Some(delay),
                    ..
                }) => {
                    tokio::time::sleep(delay).await;
                    checkout
                        .resume(ResumeValue::Return(MontyObject::none()), &mut on_print)
                        .await
                }
                // No file system, environment or clock beyond what Monty provides itself.
                Ok(TurnEvent::OsCall { .. }) => {
                    checkout
                        .resume(ResumeValue::NotHandled, &mut on_print)
                        .await
                }
                Ok(TurnEvent::NameLookup { .. }) => {
                    checkout
                        .resume_name_lookup(NameLookupResult::Undefined, &mut on_print)
                        .await
                }
                Err(PoolError::Runtime(exception)) => break exception_error_text(&exception),
                Err(err) => break Some(format!("Python sandbox failed: {err}")),
            };
        };
        drop(on_print);
        finish_checkout(checkout).await;
        result
    }

    /// The globals a script starts with: `tools`, `ALL_TOOLS` and the helpers.
    fn inputs(&self) -> NamedValues {
        let mut names: Vec<&String> = self.tools.keys().collect();
        names.sort();
        let tool_functions = names.iter().map(|name| {
            (
                MontyObject::string(name.as_str()),
                MontyObject::async_function(format!("{TOOLS_NAMESPACE}.{name}"), None),
            )
        });
        let tools_module = MontyObject::module(TOOLS_NAMESPACE, TOOLS_MODULE_ID, tool_functions);
        let all_tools = MontyObject::list(names.iter().map(|name| {
            let entry = &self.tools[name.as_str()];
            MontyObject::dict([
                (
                    MontyObject::string("name"),
                    MontyObject::string(name.as_str()),
                ),
                (
                    MontyObject::string("description"),
                    MontyObject::string(entry.definition.description.as_str()),
                ),
            ])
        }));
        let mut inputs = vec![
            (TOOLS_NAMESPACE.to_string(), tools_module),
            ("ALL_TOOLS".to_string(), all_tools),
        ];
        for helper in [
            HELPER_TEXT,
            HELPER_IMAGE,
            HELPER_STORE,
            HELPER_LOAD,
            HELPER_NOTIFY,
            HELPER_YIELD,
        ] {
            inputs.push((helper.to_string(), MontyObject::function(helper, None)));
        }
        NamedValues::from(inputs)
    }

    async fn call(&mut self, function_name: &str, args: CallArgs, call_id: u32) -> ResumeValue {
        if let Some(tool) = function_name
            .strip_prefix(TOOLS_NAMESPACE)
            .and_then(|rest| rest.strip_prefix('.'))
        {
            return self.start_tool(tool, &args, call_id);
        }
        match self.call_helper(function_name, &args).await {
            Ok(value) => ResumeValue::Return(value),
            Err(exception) => ResumeValue::Error(exception),
        }
    }

    fn start_tool(&mut self, tool: &str, args: &CallArgs, call_id: u32) -> ResumeValue {
        let Some(entry) = self.tools.get(tool) else {
            return ResumeValue::Error(MontyException::new(
                ExcType::AttributeError,
                Some(format!("tool `{tool}` is not available")),
            ));
        };
        let input = match tool_input(entry.definition.kind, args) {
            Ok(input) => input,
            Err(message) => {
                return ResumeValue::Error(MontyException::new(
                    ExcType::TypeError,
                    Some(format!("tools.{tool}: {message}")),
                ));
            }
        };
        self.next_runtime_call += 1;
        let invocation = CodeModeNestedToolCall {
            cell_id: self.context.cell_id.clone(),
            runtime_tool_call_id: format!("tool-{}", self.next_runtime_call),
            tool_name: entry.definition.tool_name.clone(),
            tool_kind: entry.definition.kind,
            input,
        };
        let delegate = Arc::clone(&self.context.delegate);
        let cancel = self.context.cancel.child_token();
        self.in_flight.spawn(async move {
            let answer = match delegate.invoke_tool(invocation, cancel).await {
                Ok(value) => ResumeValue::Return(json_to_monty(&value)),
                Err(message) => {
                    ResumeValue::Error(MontyException::new(ExcType::RuntimeError, Some(message)))
                }
            };
            (call_id, answer)
        });
        ResumeValue::Future
    }

    /// Waits until at least one of `pending` has a result, then returns all that do.
    async fn settle(&mut self, pending: &[u32]) -> Result<Vec<(u32, ResumeValue)>, String> {
        loop {
            let ready: Vec<(u32, ResumeValue)> = pending
                .iter()
                .filter_map(|id| self.finished.remove(id).map(|value| (*id, value)))
                .collect();
            if !ready.is_empty() {
                return Ok(ready);
            }
            match self.in_flight.join_next().await {
                Some(Ok((id, value))) => {
                    self.finished.insert(id, value);
                }
                Some(Err(err)) => return Err(format!("nested tool call failed: {err}")),
                None => {
                    return Err(format!(
                        "the script awaited calls {pending:?} that were never started"
                    ));
                }
            }
        }
    }

    async fn call_helper(
        &mut self,
        name: &str,
        args: &CallArgs,
    ) -> Result<MontyObject, MontyException> {
        let first = args.arg(0);
        match name {
            HELPER_TEXT => {
                let text = first.map(display_text).unwrap_or_default();
                self.context
                    .shared
                    .push_item(FunctionCallOutputContentItem::InputText { text });
            }
            HELPER_IMAGE => {
                let detail = args.arg(1).or_else(|| args.kwarg("detail"));
                let item = image_item(first, detail).map_err(type_error)?;
                self.context.shared.push_item(item);
            }
            HELPER_STORE => {
                let key = first
                    .and_then(|key| key.as_str())
                    .ok_or_else(|| type_error("store() key must be a str"))?
                    .to_string();
                let value = args
                    .arg(1)
                    .or_else(|| args.kwarg("value"))
                    .ok_or_else(|| type_error("store() needs a value"))?;
                let value =
                    monty_to_json(value).map_err(|err| type_error(format!("store(): {err}")))?;
                self.context
                    .stored
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .insert(key, value);
            }
            HELPER_LOAD => {
                let key = first
                    .and_then(|key| key.as_str())
                    .ok_or_else(|| type_error("load() key must be a str"))?;
                let stored = self
                    .context
                    .stored
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .get(key)
                    .map(json_to_monty);
                return Ok(stored.unwrap_or_else(MontyObject::none));
            }
            HELPER_NOTIFY => {
                let text = first.map(display_text).unwrap_or_default();
                self.context
                    .delegate
                    .notify(
                        self.context.tool_call_id.clone(),
                        self.context.cell_id.clone(),
                        text,
                        self.context.cancel.child_token(),
                    )
                    .await
                    .map_err(|message| MontyException::new(ExcType::RuntimeError, Some(message)))?;
            }
            HELPER_YIELD => self
                .context
                .shared
                .update(|state| state.yield_requested = true),
            _ => {
                return Err(MontyException::new(
                    ExcType::NameError,
                    Some(format!("name '{name}' is not defined")),
                ));
            }
        }
        Ok(MontyObject::none())
    }
}

/// Converts call arguments into a tool's JSON input: keyword arguments become an object,
/// a single positional argument is passed through.
fn tool_input(kind: CodeModeToolKind, args: &CallArgs) -> Result<Option<JsonValue>, String> {
    let positional = args.args().len();
    let keywords = args.kwargs().len();
    match (positional, keywords) {
        (0, 0) => Ok(match kind {
            CodeModeToolKind::Function => Some(JsonValue::Object(Default::default())),
            CodeModeToolKind::Freeform => None,
        }),
        (1, 0) => {
            let value = args.arg(0).map(monty_to_json).transpose()?;
            if kind == CodeModeToolKind::Freeform && !matches!(value, Some(JsonValue::String(_))) {
                return Err("expects a single str argument".to_string());
            }
            Ok(value)
        }
        (0, _) => {
            let mut map = serde_json::Map::new();
            for (key, value) in args.kwargs() {
                let key = key
                    .as_str()
                    .ok_or_else(|| "keyword names must be str".to_string())?;
                map.insert(key.to_string(), monty_to_json(value)?);
            }
            if kind == CodeModeToolKind::Freeform {
                return match map.remove("input") {
                    Some(JsonValue::String(text)) if map.is_empty() => {
                        Ok(Some(JsonValue::String(text)))
                    }
                    _ => Err("expects a single str argument".to_string()),
                };
            }
            Ok(Some(JsonValue::Object(map)))
        }
        _ => Err("pass either keyword arguments or a single dict, not both".to_string()),
    }
}

fn image_item(
    value: Option<monty_types::ObjectRef<'_>>,
    detail: Option<monty_types::ObjectRef<'_>>,
) -> Result<FunctionCallOutputContentItem, String> {
    let value = value.ok_or("image() needs an image URL or item")?;
    let json = monty_to_json(value)?;
    let (image_url, embedded_detail) = match &json {
        JsonValue::String(url) => (url.clone(), None),
        JsonValue::Object(map) => {
            if let Some(url) = map.get("image_url").and_then(JsonValue::as_str) {
                (
                    url.to_string(),
                    map.get("detail").and_then(JsonValue::as_str),
                )
            } else if map.get("type").and_then(JsonValue::as_str) == Some("image") {
                let data = map
                    .get("data")
                    .and_then(JsonValue::as_str)
                    .ok_or("image block is missing `data`")?;
                let mime_type = map
                    .get("mimeType")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("image/png");
                (format!("data:{mime_type};base64,{data}"), None)
            } else {
                return Err(
                    "image() expects a URL, {\"image_url\": ...} or an MCP image block".into(),
                );
            }
        }
        _ => {
            return Err("image() expects a URL, {\"image_url\": ...} or an MCP image block".into());
        }
    };
    if !image_url.starts_with("data:") {
        return Err("image() only accepts base64 `data:` URLs".into());
    }
    let detail_text = match detail {
        Some(detail) if detail.type_name() != "NoneType" => Some(
            detail
                .as_str()
                .ok_or("image() detail must be a str")?
                .to_string(),
        ),
        _ => embedded_detail.map(str::to_string),
    };
    let detail = detail_text.map(|text| parse_detail(&text)).transpose()?;
    Ok(FunctionCallOutputContentItem::InputImage { image_url, detail })
}

fn parse_detail(text: &str) -> Result<ImageDetail, String> {
    match text {
        "auto" => Ok(ImageDetail::Auto),
        "low" => Ok(ImageDetail::Low),
        "high" => Ok(ImageDetail::High),
        "original" => Ok(ImageDetail::Original),
        other => Err(format!(
            "image() detail must be \"auto\", \"low\", \"high\" or \"original\", got {other:?}"
        )),
    }
}

fn type_error(message: impl Into<String>) -> MontyException {
    MontyException::new(ExcType::TypeError, Some(message.into()))
}

/// `None` for a clean `SystemExit`, otherwise the traceback.
fn exception_error_text(exception: &MontyException) -> Option<String> {
    if exception.exc_type() == ExcType::SystemExit
        && matches!(exception.message(), None | Some("None" | "0"))
    {
        return None;
    }
    Some(exception.to_string())
}

fn ensure_newline(mut text: String) -> String {
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text
}

async fn finish_checkout(checkout: Checkout) {
    if let Err(err) = checkout.finish().await {
        // The worker is discarded and replaced; the cell's result stands.
        tracing::debug!("monty checkout did not return cleanly: {err}");
    }
}
