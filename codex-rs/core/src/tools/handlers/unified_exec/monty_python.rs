//! Diverts `exec_command` calls that only run an inline Python script to the Monty pool.
//!
//! The model habitually writes `python3 - <<'PY' ... PY` for small computations. Running
//! that script in a prewarmed Monty worker skips the shell and CPython startup and keeps
//! the script inside a sandbox that sees only the working directory. When Monty cannot run
//! the script (no worker binary, a pool failure, or a feature Monty lacks) the command is
//! handed back to the shell untouched, so nothing the model could do before is lost.

use std::path::Path;
use std::sync::Arc;

use codex_features::Feature;
use codex_protocol::exec_output::ExecToolCallOutput;
use codex_protocol::exec_output::StreamOutput;
use codex_protocol::protocol::ExecCommandSource;
use codex_utils_output_truncation::approx_token_count;

use super::preparation::PreparedExecCommand;
use crate::monty::DEFAULT_TIMEOUT;
use crate::monty::MontyRequest;
use crate::monty::python_command::PythonSource;
use crate::monty::python_command::python_source_from_command;
use crate::monty::resolve_binary;
use crate::monty::run_python;
use crate::sandboxing::SandboxPermissions;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use crate::tools::context::ExecCommandToolOutput;
use crate::tools::context::SharedTurnDiffTracker;
use crate::tools::events::ToolEmitter;
use crate::tools::events::ToolEventCtx;
use crate::tools::events::ToolEventStage;
use crate::tools::handlers::run_python::cwd_is_writable;
use crate::unified_exec::generate_chunk_id;
use crate::unified_exec::output::CapturedStream;

/// Runs the command's Python in Monty when it is nothing but an inline Python script.
/// `None` means the shell should run the command as usual.
pub(super) async fn intercept_monty_python(
    prepared: &PreparedExecCommand,
    native_cwd: Option<&Path>,
    session: &Arc<Session>,
    step_context: &Arc<StepContext>,
    tracker: &SharedTurnDiffTracker,
    call_id: &str,
) -> Option<ExecCommandToolOutput> {
    if !session.enabled(Feature::MontyPython)
        || prepared.turn_environment.environment.is_remote()
        // Escalation or extra permissions mean the model wants the host, not a sandbox.
        || !matches!(
            prepared.permissions.sandbox_permissions,
            SandboxPermissions::UseDefault
        )
    {
        return None;
    }
    let source = python_source_from_command(&prepared.resolved.command)?;
    let cwd = native_cwd?;
    let Some(binary) = resolve_binary(&step_context.turn.config) else {
        tracing::debug!("python command left to the shell: no monty binary configured");
        return None;
    };
    let code = match source {
        PythonSource::Inline(code) => code,
        PythonSource::File(path) => match tokio::fs::read_to_string(cwd.join(&path)).await {
            Ok(code) => code,
            Err(err) => {
                tracing::debug!("python command left to the shell: cannot read `{path}`: {err}");
                return None;
            }
        },
    };
    let writable = cwd_is_writable(&prepared.turn_environment, cwd);
    let output = match run_python(
        binary,
        MontyRequest {
            code: &code,
            cwd,
            writable,
            timeout: DEFAULT_TIMEOUT,
        },
    )
    .await
    {
        Ok(output) => output,
        Err(err) => {
            tracing::warn!("python command left to the shell: {err}");
            return None;
        }
    };
    if output.likely_unsupported() {
        let exception = output
            .exception
            .as_ref()
            .map(|exc| {
                exc.to_string()
                    .lines()
                    .last()
                    .unwrap_or_default()
                    .to_owned()
            })
            .unwrap_or_default();
        tracing::info!("python command left to the shell: monty raised {exception}");
        return None;
    }

    let mut text = output.render();
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&format!(
        "[monty] ran in the Monty sandbox in {:.1?}\n",
        output.duration
    ));

    let emitter = ToolEmitter::unified_exec(
        &prepared.resolved.command,
        prepared.cwd.clone(),
        ExecCommandSource::UnifiedExecStartup,
        /*process_id*/ None,
        /*plugin_attribution*/ None,
    );
    let ctx = ToolEventCtx::new(
        session.as_ref(),
        step_context.turn.as_ref(),
        &step_context.settings.model_info,
        call_id,
        Some(tracker),
    );
    emitter.emit(ctx, ToolEventStage::Begin).await;
    emitter
        .emit(
            ctx,
            ToolEventStage::Success {
                output: ExecToolCallOutput {
                    exit_code: output.exit_code(),
                    stdout: StreamOutput::new(text.clone()),
                    stderr: StreamOutput::new(String::new()),
                    aggregated_output: StreamOutput::new(text.clone()),
                    duration: output.duration,
                    timed_out: false,
                },
                applied_patch_delta: None,
            },
        )
        .await;

    Some(ExecCommandToolOutput {
        event_call_id: call_id.to_string(),
        chunk_id: generate_chunk_id(),
        wall_time: output.duration,
        raw_output: CapturedStream {
            text: text.clone(),
            omitted_bytes: 0,
        },
        truncation_policy: step_context.settings.model_info.truncation_policy.into(),
        max_output_tokens: prepared.max_output_tokens,
        process_id: None,
        exit_code: Some(output.exit_code()),
        original_token_count: Some(approx_token_count(&text)),
        hook_command: Some(prepared.hook_command.clone()),
        tool_cwd: Some(prepared.cwd.clone()),
    })
}
