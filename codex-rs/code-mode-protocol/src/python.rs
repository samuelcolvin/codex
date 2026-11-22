//! The Python flavour of Code Mode, run by Monty.
//!
//! The JavaScript flavour documents nested tools as TypeScript declarations. This module
//! documents them as Python stubs instead and builds the `exec` description that goes with
//! them. Everything else in the protocol (cells, `wait`, tool dispatch) is shared.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use codex_protocol::ToolName;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value as JsonValue;

use crate::description::CodeModeToolKind;
use crate::description::DeferredToolDiscovery;
use crate::description::ExecDefinition;
use crate::description::ExecDefinitions;
use crate::description::ImageDetailVisibility;
use crate::description::ToolDefinition;
use crate::description::ToolNamespaceDescription;
use crate::json_schema_types::DEFAULT_INPUT_SCHEMA_MAX_BYTES;

/// The language the model writes `exec` cells in.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CodeModeLanguage {
    #[default]
    JavaScript,
    Python,
}

/// First-line pragma that sets `yield_time_ms` and `max_output_tokens` for a Python cell.
pub const PYTHON_CODE_MODE_PRAGMA_PREFIX: &str = "# @exec:";

/// Name of the namespace object nested tools hang off, in both languages.
pub const TOOLS_NAMESPACE: &str = "tools";

/// Nested object types deeper than this render as `Any`.
const MAX_TYPE_DEPTH: usize = 6;
/// Local `$ref`s followed while rendering one type before giving up with `Any`.
const MAX_REF_EXPANSIONS: usize = 16;
/// Parameter comments longer than this are cut.
const MAX_PARAM_COMMENT_CHARS: usize = 240;

const PYTHON_KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
];

const PYTHON_EXEC_DESCRIPTION_TEMPLATE: &str = r#"Run Python code to orchestrate/compose tool calls
- Runs the provided code as a fresh script in Monty, a sandboxed Python interpreter written in Rust. Top-level `await` works.
- All nested tools are async functions on the global `tools` namespace, for example `await tools.exec_command(cmd="ls")`. Tool names are normalized to Python identifiers, for example `await tools.mcp__ologs__get_profile(...)`.
- Call a nested tool with keyword arguments, with a single `dict` of arguments, or with a single `str` for tools that take free-form input. Tools return a `dict`, `list` or `str`, as their declaration says.
- Run independent tool calls concurrently with `await asyncio.gather(...)` after `import asyncio`.
- Monty implements most of the Python language and a subset of the standard library (for example `asyncio`, `json`, `re`, `math`, `datetime`, `dataclasses`, `typing`). There is no file system, network, subprocess, environment or third-party package access except through tools.
- Accepts raw Python source text, not JSON, quoted strings, or markdown code fences.
- You may optionally start the tool input with a first-line pragma like `# @exec: {"yield_time_ms": 10000, "max_output_tokens": 1000}`.
- `yield_time_ms` asks `exec` to yield early if the script is still running. Defaults to {{ default_exec_yield_time_ms }} ms.
- `max_output_tokens` sets the token budget for direct `exec` results. Defaults to 10000 tokens.
- `print()` output is returned to you. If the script ends with an expression whose value is not `None`, its `repr()` is returned as well.
- Variables do not survive between `exec` calls; use `store()` and `load()` to keep values. Tool calls still running when the script ends are cancelled.
- In declarations below, dict shapes are written `{key: type, optional_key?: type}`."#;

const PYTHON_EXEC_HELPERS_TEMPLATE: &str = r#"- Global helpers:
- `text(value: str | int | float | bool | None)`: Appends a text item. Non-string values are converted with `json.dumps(...)` when possible.
- {{ image_helper }}
- `store(key: str, value: Any)`: stores a JSON-serializable value under a string key for later `exec` calls in the same session.
- `load(key: str)`: returns the stored value for a string key, or `None` if it is missing.
- `notify(value: str | int | float | bool | None)`: immediately injects an extra `custom_tool_call_output` for the current `exec` call. Values are converted like `text(...)`.
- `yield_control()`: yields the accumulated output to the model immediately while the script keeps running.
- `ALL_TOOLS`: metadata for the enabled nested tools as a list of `{"name": str, "description": str}` dicts.
- `raise SystemExit` ends the script early; `SystemExit()`, `SystemExit(0)` and `SystemExit(None)` count as success."#;

const LEGACY_IMAGE_HELPER: &str = r#"`image(image_url_or_item: str | dict, detail: Literal["auto", "low", "high", "original"] | None = None)`: Appends an image item. `image_url` should be a base64-encoded `data:` URL. To forward an MCP tool image, pass an individual image block from `result["content"]`, for example `image(result["content"][0])`. The `detail` argument overrides any detail in the first argument."#;
const UNIFIED_IMAGE_HELPER: &str = r#"`image(image_url_or_item: str | dict)`: Appends an image item. `image_url` should be a base64-encoded `data:` URL. To forward an MCP tool image, pass an individual image block from `result["content"]`, for example `image(result["content"][0])`."#;

const DEFERRED_NESTED_TOOLS_GUIDANCE: &str = r#"Some deferred nested tools may be omitted from this description. They are still available on the global `tools` namespace and listed in `ALL_TOOLS`.
To find one, filter `ALL_TOOLS` by `"name"` and `"description"`.

Tool availability can change between calls."#;
const TOOL_SEARCH_GUIDANCE: &str = r#"Use `await tools.tool_search(query="...", limit=8)` for BM25-ranked discovery of deferred tools (up to `limit` matches; 8 by default). It returns `{"tools": [{"name": ..., "description": ...}]}` with callable names and full declarations. Print the results and read unfamiliar declarations before calling `getattr(tools, result["name"])(...)` in a later execution."#;

const MCP_PYTHON_PREAMBLE: &str = r#"# MCP tool results are dicts of this shape (TypedDict notation; `?` marks optional keys).
ContentBlock = (
    {type: Literal["text"], text: str}
    | {type: Literal["image"], data: str, mimeType: str}
    | {type: Literal["audio"], data: str, mimeType: str}
    | {type: Literal["resource_link"], name: str, uri: str, title?: str, description?: str, mimeType?: str}
    | {type: Literal["resource"], resource: {uri: str, mimeType?: str, text?: str, blob?: str}}
)
CallToolResult = {content: list[ContentBlock], isError?: bool, structuredContent?: Any, _meta?: dict[str, Any]}"#;

/// Turns a tool's code-mode name into the Python attribute it is exposed as on `tools`.
pub fn python_identifier(tool_key: &str) -> String {
    let mut identifier: String = tool_key
        .chars()
        .enumerate()
        .map(|(index, ch)| {
            let valid = ch == '_' || ch.is_ascii_alphabetic() || (index > 0 && ch.is_ascii_digit());
            if valid { ch } else { '_' }
        })
        .collect();
    if identifier.is_empty() {
        identifier.push('_');
    }
    if PYTHON_KEYWORDS.contains(&identifier.as_str()) {
        identifier.push('_');
    }
    identifier
}

fn is_python_identifier(name: &str) -> bool {
    python_identifier(name) == name
}

/// Appends the declaration of a nested tool, in `language`, to its description.
pub fn augment_tool_definition_for_language(
    definition: ToolDefinition,
    language: CodeModeLanguage,
) -> ToolDefinition {
    match language {
        CodeModeLanguage::JavaScript => crate::description::augment_tool_definition(definition),
        CodeModeLanguage::Python => augment_python_tool_definition(definition),
    }
}

/// Appends the Python declaration of a nested tool to its description.
pub fn augment_python_tool_definition(mut definition: ToolDefinition) -> ToolDefinition {
    if definition.name != crate::PUBLIC_TOOL_NAME {
        definition.description = render_python_sample_for_definition(&definition);
    }
    definition
}

fn render_python_sample_for_definition(definition: &ToolDefinition) -> String {
    let stub = render_python_tool_stub(definition);
    format!(
        "{}\n\nexec tool declaration:\n```python\n{stub}\n```",
        definition.description
    )
}

/// Renders one nested tool as an `async def` stub on the `tools` namespace.
pub fn render_python_tool_stub(definition: &ToolDefinition) -> String {
    let name = python_identifier(&definition.name);
    let output_type = render_output_type(definition.output_schema.as_ref());
    let budget = definition
        .input_schema_max_bytes
        .unwrap_or(DEFAULT_INPUT_SCHEMA_MAX_BYTES);
    let params = match definition.kind {
        CodeModeToolKind::Freeform => "input: str, /".to_string(),
        CodeModeToolKind::Function => match definition.input_schema.as_ref() {
            Some(schema) => {
                let rendered = render_function_params(schema);
                if rendered.len() > budget {
                    "**kwargs: Any".to_string()
                } else {
                    rendered
                }
            }
            None => "**kwargs: Any".to_string(),
        },
    };
    format!("async def {TOOLS_NAMESPACE}.{name}({params}) -> {output_type}: ...")
}

/// Keyword-only parameters for an object schema, or one positional dict when the property
/// names are not all valid Python identifiers.
fn render_function_params(schema: &JsonValue) -> String {
    let mut renderer = PythonTypeRenderer::new(schema);
    let Some(map) = schema.as_object() else {
        return "**kwargs: Any".to_string();
    };
    let properties = map.get("properties").and_then(JsonValue::as_object);
    let required: BTreeSet<&str> = map
        .get("required")
        .and_then(JsonValue::as_array)
        .map(|values| values.iter().filter_map(JsonValue::as_str).collect())
        .unwrap_or_default();

    let Some(properties) = properties.filter(|properties| !properties.is_empty()) else {
        return match map.get("additionalProperties") {
            Some(JsonValue::Bool(false)) => String::new(),
            Some(value @ JsonValue::Object(_)) => {
                format!("**kwargs: {}", renderer.render(value, 0))
            }
            _ => "**kwargs: Any".to_string(),
        };
    };

    if !properties.keys().all(|key| is_python_identifier(key)) {
        return format!("args: {}, /", renderer.render(schema, 0));
    }

    // Required parameters first so the stub reads like a real signature.
    let mut ordered: Vec<(&String, &JsonValue)> = properties.iter().collect();
    ordered.sort_by_key(|(key, _)| !required.contains(key.as_str()));

    let mut lines = vec!["*,".to_string()];
    for (key, property) in ordered {
        let property_type = renderer.render(property, 1);
        let mut line = if required.contains(key.as_str()) {
            format!("{key}: {property_type},")
        } else if property_type == "Any" || property_type.ends_with("| None") {
            format!("{key}: {property_type} = None,")
        } else {
            format!("{key}: {property_type} | None = None,")
        };
        if let Some(description) = property
            .get("description")
            .and_then(JsonValue::as_str)
            .map(single_line_comment)
            .filter(|description| !description.is_empty())
        {
            line.push_str("  # ");
            line.push_str(&description);
        }
        lines.push(line);
    }
    let body = lines
        .iter()
        .map(|line| format!("    {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("\n{body}\n")
}

fn single_line_comment(text: &str) -> String {
    let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if joined.chars().count() > MAX_PARAM_COMMENT_CHARS {
        let cut: String = joined.chars().take(MAX_PARAM_COMMENT_CHARS).collect();
        format!("{cut}...")
    } else {
        joined
    }
}

fn render_output_type(output_schema: Option<&JsonValue>) -> String {
    let Some(schema) = output_schema else {
        return "Any".to_string();
    };
    if is_mcp_call_tool_result(schema) {
        return "CallToolResult".to_string();
    }
    PythonTypeRenderer::new(schema).render(schema, 0)
}

fn is_mcp_call_tool_result(schema: &JsonValue) -> bool {
    let Some(properties) = schema.get("properties").and_then(JsonValue::as_object) else {
        return false;
    };
    let is_type = |key: &str, expected: &str| {
        properties
            .get(key)
            .and_then(|value| value.get("type"))
            .and_then(JsonValue::as_str)
            == Some(expected)
    };
    is_type("content", "array") && is_type("isError", "boolean")
}

/// Renders JSON Schema as compact Python type expressions. Object shapes use
/// `{key: type, optional?: type}`, which is not valid Python but reads unambiguously.
struct PythonTypeRenderer<'a> {
    root: &'a JsonValue,
    remaining_ref_expansions: usize,
}

impl<'a> PythonTypeRenderer<'a> {
    fn new(root: &'a JsonValue) -> Self {
        Self {
            root,
            remaining_ref_expansions: MAX_REF_EXPANSIONS,
        }
    }

    fn render(&mut self, schema: &JsonValue, depth: usize) -> String {
        if depth > MAX_TYPE_DEPTH {
            return "Any".to_string();
        }
        match schema {
            JsonValue::Object(map) => self.render_map(map, depth),
            _ => "Any".to_string(),
        }
    }

    fn render_map(&mut self, map: &serde_json::Map<String, JsonValue>, depth: usize) -> String {
        if let Some(reference) = map.get("$ref").and_then(JsonValue::as_str) {
            let Some(pointer) = reference.strip_prefix('#') else {
                return "Any".to_string();
            };
            if self.remaining_ref_expansions == 0 {
                return "Any".to_string();
            }
            self.remaining_ref_expansions -= 1;
            let root = self.root;
            let target = if pointer.is_empty() {
                Some(root)
            } else {
                root.pointer(pointer)
            };
            return match target {
                Some(target) => self.render(target, depth + 1),
                None => "Any".to_string(),
            };
        }
        if let Some(value) = map.get("const") {
            return format!("Literal[{}]", python_literal(value));
        }
        if let Some(values) = map.get("enum").and_then(JsonValue::as_array)
            && !values.is_empty()
        {
            let literals = values.iter().map(python_literal).collect::<Vec<_>>();
            return format!("Literal[{}]", literals.join(", "));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(variants) = map.get(key).and_then(JsonValue::as_array)
                && !variants.is_empty()
            {
                let rendered = variants
                    .iter()
                    .map(|variant| self.render(variant, depth))
                    .collect::<Vec<_>>();
                return union(rendered);
            }
        }
        if let Some(variants) = map.get("allOf").and_then(JsonValue::as_array)
            && let Some(first) = variants.first()
        {
            return if variants.len() == 1 {
                self.render(first, depth)
            } else {
                "Any".to_string()
            };
        }
        match map.get("type") {
            Some(JsonValue::String(schema_type)) => self.render_type(map, schema_type, depth),
            Some(JsonValue::Array(types)) => {
                let rendered = types
                    .iter()
                    .filter_map(JsonValue::as_str)
                    .map(|schema_type| self.render_type(map, schema_type, depth))
                    .collect::<Vec<_>>();
                if rendered.is_empty() {
                    "Any".to_string()
                } else {
                    union(rendered)
                }
            }
            _ if map.contains_key("properties") => self.render_object(map, depth),
            _ if map.contains_key("items") => self.render_array(map, depth),
            _ => "Any".to_string(),
        }
    }

    fn render_type(
        &mut self,
        map: &serde_json::Map<String, JsonValue>,
        schema_type: &str,
        depth: usize,
    ) -> String {
        match schema_type {
            "string" => "str".to_string(),
            "integer" => "int".to_string(),
            "number" => "float".to_string(),
            "boolean" => "bool".to_string(),
            "null" => "None".to_string(),
            "array" => self.render_array(map, depth),
            "object" => self.render_object(map, depth),
            _ => "Any".to_string(),
        }
    }

    fn render_array(&mut self, map: &serde_json::Map<String, JsonValue>, depth: usize) -> String {
        match map.get("items") {
            Some(items @ JsonValue::Object(_)) => {
                format!("list[{}]", self.render(items, depth + 1))
            }
            _ => "list[Any]".to_string(),
        }
    }

    fn render_object(&mut self, map: &serde_json::Map<String, JsonValue>, depth: usize) -> String {
        let properties = map
            .get("properties")
            .and_then(JsonValue::as_object)
            .filter(|properties| !properties.is_empty());
        let Some(properties) = properties else {
            return match map.get("additionalProperties") {
                Some(value @ JsonValue::Object(_)) => {
                    format!("dict[str, {}]", self.render(value, depth + 1))
                }
                _ => "dict[str, Any]".to_string(),
            };
        };
        let required: BTreeSet<&str> = map
            .get("required")
            .and_then(JsonValue::as_array)
            .map(|values| values.iter().filter_map(JsonValue::as_str).collect())
            .unwrap_or_default();
        let fields = properties
            .iter()
            .map(|(key, value)| {
                let optional = if required.contains(key.as_str()) {
                    ""
                } else {
                    "?"
                };
                format!("{key}{optional}: {}", self.render(value, depth + 1))
            })
            .collect::<Vec<_>>();
        format!("{{{}}}", fields.join(", "))
    }
}

fn union(variants: Vec<String>) -> String {
    let mut seen = BTreeSet::new();
    let mut unique = Vec::new();
    for variant in variants {
        if variant == "Any" {
            return "Any".to_string();
        }
        if seen.insert(variant.clone()) {
            unique.push(variant);
        }
    }
    // Keep `None` last so optional types read as `T | None`.
    unique.sort_by_key(|variant| variant == "None");
    unique.join(" | ")
}

fn python_literal(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "None".to_string(),
        JsonValue::Bool(true) => "True".to_string(),
        JsonValue::Bool(false) => "False".to_string(),
        JsonValue::Number(number) => number.to_string(),
        JsonValue::String(text) => serde_json::to_string(text).unwrap_or_else(|_| "...".into()),
        JsonValue::Array(_) | JsonValue::Object(_) => "...".to_string(),
    }
}

/// Builds the Python `exec` description. Takes the same inputs as the JavaScript builder,
/// minus model-supplied message overrides, which are written for JavaScript.
#[allow(clippy::too_many_arguments)]
pub fn build_python_exec_tool_description_and_definitions(
    enabled_tools: &[ToolDefinition],
    namespace_descriptions: &BTreeMap<String, ToolNamespaceDescription>,
    default_exec_yield_time_ms: u64,
    code_mode_only: bool,
    image_detail_visibility: ImageDetailVisibility,
    deferred_tool_discovery: DeferredToolDiscovery,
    incremental_exec_enabled: bool,
) -> (String, ExecDefinitions) {
    let image_helper = match image_detail_visibility {
        ImageDetailVisibility::Visible => LEGACY_IMAGE_HELPER,
        ImageDetailVisibility::Hidden => UNIFIED_IMAGE_HELPER,
    };
    let yield_time = default_exec_yield_time_ms.to_string();
    let render = |template: &str| {
        template
            .replace("{{ default_exec_yield_time_ms }}", &yield_time)
            .replace("{{ image_helper }}", image_helper)
    };

    let mut definitions = ExecDefinitions::default();
    let mut sections = vec![render(PYTHON_EXEC_DESCRIPTION_TEMPLATE)];
    let helpers = render(PYTHON_EXEC_HELPERS_TEMPLATE);
    if incremental_exec_enabled {
        definitions.helper_signatures = helpers
            .lines()
            .filter_map(|line| {
                line.strip_prefix("- `")?
                    .split_once("`:")
                    .map(|(signature, _)| signature)
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
    sections.push(helpers);
    sections.push(match deferred_tool_discovery {
        DeferredToolDiscovery::RankedSearch if !code_mode_only => {
            format!("{DEFERRED_NESTED_TOOLS_GUIDANCE}\n\n{TOOL_SEARCH_GUIDANCE}")
        }
        DeferredToolDiscovery::Catalog | DeferredToolDiscovery::RankedSearch => {
            DEFERRED_NESTED_TOOLS_GUIDANCE.to_string()
        }
    });
    if !code_mode_only {
        return (sections.join("\n\n"), definitions);
    }

    sections.push(format!(
        "Shared MCP Types:\n```python\n{MCP_PYTHON_PREAMBLE}\n```"
    ));
    if enabled_tools.is_empty() {
        return (sections.join("\n\n"), definitions);
    }

    let mut current_namespace: Option<&str> = None;
    let mut seen_namespaces = BTreeSet::new();
    let mut tool_sections = Vec::with_capacity(enabled_tools.len());
    for tool in enabled_tools {
        let namespace_description = tool
            .tool_name
            .namespace
            .as_ref()
            .and_then(|namespace| namespace_descriptions.get(namespace));
        let next_namespace = namespace_description.map(|description| description.name.as_str());
        if let Some(namespace_description) = namespace_description {
            let text = namespace_description.description.trim();
            if !text.is_empty() {
                let new_entry =
                    incremental_exec_enabled && seen_namespaces.insert(&namespace_description.name);
                let new_group = next_namespace != current_namespace;
                let description = format!("## {}\n{text}", namespace_description.name);
                if new_entry {
                    definitions.entries.push(ExecDefinition {
                        key: format!("Namespace instructions: {}", namespace_description.name),
                        description: description.clone(),
                    });
                }
                if new_group {
                    tool_sections.push(description);
                }
            }
        }
        current_namespace = next_namespace;

        let global_name = python_identifier(&tool.name);
        let heading = if global_name == tool.name {
            format!("### `{global_name}`")
        } else {
            format!("### `{global_name}` (`{}`)", tool.name)
        };
        let sample = render_python_sample_for_definition(tool);
        let description = format!("{heading}\n{}", sample.trim());
        if incremental_exec_enabled {
            definitions.entries.push(ExecDefinition {
                key: format!("Tool: tools.{global_name}"),
                description: description.clone(),
            });
        }
        tool_sections.push(description);
    }
    sections.push(tool_sections.join("\n\n"));
    (sections.join("\n\n"), definitions)
}

/// Finds the tool definition a Python attribute name refers to.
pub fn python_tool_lookup(
    tools: &[ToolDefinition],
) -> BTreeMap<String, (ToolName, CodeModeToolKind)> {
    let mut lookup = BTreeMap::new();
    for tool in tools {
        lookup
            .entry(python_identifier(&tool.name))
            .or_insert_with(|| (tool.tool_name.clone(), tool.kind));
    }
    lookup
}

#[cfg(test)]
#[path = "python_tests.rs"]
mod tests;
