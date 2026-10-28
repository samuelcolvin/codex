//! `run_python`: runs a Python script in the Monty sandbox with the working directory
//! mounted. See [`crate::monty`].

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use codex_protocol::exec_output::ExecToolCallOutput;
use codex_protocol::exec_output::StreamOutput;
use codex_protocol::protocol::ExecCommandSource;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use serde::Deserialize;

use crate::function_tool::FunctionCallError;
use crate::monty::DEFAULT_TIMEOUT;
use crate::monty::MAX_TIMEOUT;
use crate::monty::MontyRequest;
use crate::monty::MontyRunOutput;
use crate::monty::resolve_binary;
use crate::monty::run_python;
use crate::session::turn_context::TurnEnvironment;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolOutput;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::events::ToolEmitter;
use crate::tools::events::ToolEventCtx;
use crate::tools::events::ToolEventStage;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::selected_executor;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExecutor;

const TOOL_NAME: &str = "run_python";

const DESCRIPTION: &str = "Runs a Python script in Monty, a sandboxed Python interpreter that starts in under a millisecond. \
Use it instead of running python through exec_command for calculations, date arithmetic, parsing and transforming data, \
and reading or writing files under the working directory.

The working directory is mounted at its real path, read/write when the sandbox policy allows writes, and is the script's cwd; \
paths outside it raise PermissionError. There is no network, no subprocess, no environment variables, no sys.argv and no third-party packages.

Monty implements a subset of Python 3.14: functions, closures, simple classes without inheritance, dataclasses, comprehensions, \
match, f-strings, try/except, with, async/await. Not supported: class inheritance, generators (yield), del, user-defined exception \
classes, imports of local modules. Available modules: asyncio, base64, binascii, bisect, collections, copy, csv, dataclasses, datetime, \
decimal, functools, hashlib, itertools, json, math, os, os.path, pathlib, random, re, sys, time, tomllib, typing, unicodedata.

Returns print() output, then the value of the final expression when it is not None, then the traceback of any uncaught exception. \
If the script needs something Monty lacks, the error says so; run it with exec_command instead.";

pub struct RunPythonHandler;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunPythonArgs {
    code: String,
    #[serde(default)]
    workdir: Option<String>,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

fn create_run_python_tool() -> ToolSpec {
    let properties = BTreeMap::from([
        (
            "code".to_string(),
            JsonSchema::string(Some("Python source to run.".to_string())),
        ),
        (
            "workdir".to_string(),
            JsonSchema::string(Some(
                "Directory to mount and run in. Defaults to the turn cwd.".to_string(),
            )),
        ),
        (
            "timeout_ms".to_string(),
            JsonSchema::number(Some(format!(
                "Execution time budget in milliseconds. Defaults to {}; capped at {}.",
                DEFAULT_TIMEOUT.as_millis(),
                MAX_TIMEOUT.as_millis()
            ))),
        ),
    ]);
    ToolSpec::Function(ResponsesApiTool {
        name: TOOL_NAME.to_string(),
        description: DESCRIPTION.to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            properties,
            Some(vec!["code".to_string()]),
            /*additional_properties*/ Some(false.into()),
        ),
        output_schema: None,
    })
}

/// Whether the turn's sandbox policy lets commands write under `cwd`.
pub(crate) fn cwd_is_writable(environment: &TurnEnvironment, cwd: &Path) -> bool {
    let policy = codex_sandboxing::compatibility_sandbox_policy_for_permission_profile(
        environment.permission_profile(),
        cwd,
    );
    policy.has_full_disk_write_access() || !policy.get_writable_roots_with_cwd(cwd).is_empty()
}

/// The model-facing text for a run: output, final value, traceback, and a hint when the
/// failure is a gap in Monty rather than in the script.
fn render_for_model(output: &MontyRunOutput) -> String {
    let mut text = output.output.clone();
    if let Some(result) = &output.result {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(result);
        text.push('\n');
    }
    if let Some(exc) = &output.exception {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&exc.to_string());
        if !text.ends_with('\n') {
            text.push('\n');
        }
        if output.likely_unsupported() {
            text.push_str(
                "\nMonty does not support something this script needs. Run it with exec_command instead.\n",
            );
        }
    }
    if text.is_empty() {
        text.push_str("(no output)\n");
    }
    text
}

impl ToolExecutor<ToolInvocation> for RunPythonHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        create_run_python_tool()
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        true
    }

    fn handle<'a>(&'a self, invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'a>
    where
        ToolInvocation: 'a,
    {
        Box::pin(handle_call(invocation))
    }
}

async fn handle_call(invocation: ToolInvocation) -> Result<Box<dyn ToolOutput>, FunctionCallError> {
    let ToolInvocation {
        environments,
        session,
        step_context,
        tracker,
        call_id,
        payload,
        ..
    } = invocation;
    let ToolPayload::Function { arguments } = payload else {
        return Err(FunctionCallError::RespondToModel(format!(
            "{TOOL_NAME} handler received unsupported payload"
        )));
    };
    let args: RunPythonArgs = parse_arguments(&arguments)?;
    if args.code.trim().is_empty() {
        return Err(FunctionCallError::RespondToModel(
            "`code` must not be empty".to_string(),
        ));
    }
    let timeout = args
        .timeout_ms
        .map_or(DEFAULT_TIMEOUT, Duration::from_millis)
        .min(MAX_TIMEOUT);

    let turn_environment = selected_executor(&environments, &step_context.environments)?;
    if turn_environment.environment.is_remote() {
        return Err(FunctionCallError::RespondToModel(format!(
            "{TOOL_NAME} only runs against a local environment; use exec_command"
        )));
    }
    let cwd_uri = match args.workdir.as_deref().filter(|dir| !dir.is_empty()) {
        Some(workdir) => turn_environment.cwd().join(workdir).map_err(|err| {
            FunctionCallError::RespondToModel(format!("invalid workdir `{workdir}`: {err}"))
        })?,
        None => turn_environment.cwd().clone(),
    };
    let cwd = cwd_uri.to_abs_path().map_err(|err| {
        FunctionCallError::RespondToModel(format!("workdir `{cwd_uri}` is not a local path: {err}"))
    })?;
    if !cwd.as_path().is_dir() {
        return Err(FunctionCallError::RespondToModel(format!(
            "workdir `{}` is not a directory",
            cwd.as_path().display()
        )));
    }
    let Some(binary) = resolve_binary(&step_context.turn.config) else {
        return Err(FunctionCallError::RespondToModel(
            "no monty worker binary: set `monty_binary` in config.toml or put `monty` on PATH"
                .to_string(),
        ));
    };
    let writable = cwd_is_writable(turn_environment, cwd.as_path());

    let output = run_python(
        binary,
        MontyRequest {
            code: &args.code,
            cwd: cwd.as_path(),
            writable,
            timeout,
        },
    )
    .await
    .map_err(|err| FunctionCallError::RespondToModel(format!("monty is unavailable: {err}")))?;

    // Surface the run like a shell command so clients show the code and its output.
    let command = vec!["monty".to_string(), "-c".to_string(), args.code.clone()];
    let emitter = ToolEmitter::unified_exec(
        &command,
        cwd_uri,
        ExecCommandSource::UnifiedExecStartup,
        /*process_id*/ None,
        /*plugin_attribution*/ None,
    );
    let ctx = ToolEventCtx::new(
        session.as_ref(),
        step_context.turn.as_ref(),
        &step_context.settings.model_info,
        &call_id,
        Some(&tracker),
    );
    emitter.emit(ctx, ToolEventStage::Begin).await;
    let rendered = output.render();
    emitter
        .emit(
            ctx,
            ToolEventStage::Success {
                output: ExecToolCallOutput {
                    exit_code: output.exit_code(),
                    stdout: StreamOutput::new(rendered.clone()),
                    stderr: StreamOutput::new(String::new()),
                    aggregated_output: StreamOutput::new(rendered),
                    duration: output.duration,
                    timed_out: false,
                },
                applied_patch_delta: None,
            },
        )
        .await;

    let success = output.exception.is_none();
    Ok(boxed_tool_output(FunctionToolOutput::from_text(
        render_for_model(&output),
        Some(success),
    )))
}

impl CoreToolRuntime for RunPythonHandler {}
