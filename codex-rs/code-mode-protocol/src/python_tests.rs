use super::*;
use pretty_assertions::assert_eq;
use serde_json::json;

fn function_tool(name: &str, input_schema: JsonValue) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        tool_name: ToolName::plain(name),
        description: format!("Runs {name}."),
        kind: CodeModeToolKind::Function,
        input_schema: Some(input_schema),
        input_schema_max_bytes: None,
        output_schema: None,
    }
}

#[test]
fn python_identifier_replaces_invalid_characters_and_keywords() {
    assert_eq!(
        python_identifier("mcp__ologs__get-profile"),
        "mcp__ologs__get_profile"
    );
    assert_eq!(python_identifier("$tool"), "_tool");
    assert_eq!(python_identifier("9lives"), "_lives");
    assert_eq!(python_identifier("import"), "import_");
    assert_eq!(python_identifier(""), "_");
}

#[test]
fn stub_uses_keyword_only_parameters_with_required_first() {
    let tool = function_tool(
        "exec_command",
        json!({
            "type": "object",
            "properties": {
                "workdir": {"type": "string", "description": "Working directory."},
                "cmd": {"type": "string", "description": "Shell command\n to run."},
                "tty": {"type": "boolean"},
                "mode": {"enum": ["fast", "slow"]},
            },
            "required": ["cmd"],
        }),
    );
    assert_eq!(
        render_python_tool_stub(&tool),
        r#"async def tools.exec_command(
    *,
    cmd: str,  # Shell command to run.
    mode: Literal["fast", "slow"] | None = None,
    tty: bool | None = None,
    workdir: str | None = None,  # Working directory.
) -> Any: ..."#
    );
}

#[test]
fn stub_falls_back_to_positional_dict_for_non_identifier_keys() {
    let tool = function_tool(
        "search",
        json!({
            "type": "object",
            "properties": {
                "max-results": {"type": "integer"},
                "items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
            },
            "required": ["max-results"],
        }),
    );
    assert_eq!(
        render_python_tool_stub(&tool),
        "async def tools.search(args: {items?: list[{id: str}], max-results: int}, /) -> Any: ..."
    );
}

#[test]
fn freeform_and_mcp_tools() {
    let freeform = ToolDefinition {
        name: "apply_patch".to_string(),
        tool_name: ToolName::plain("apply_patch"),
        description: "Patch files.".to_string(),
        kind: CodeModeToolKind::Freeform,
        input_schema: None,
        input_schema_max_bytes: None,
        output_schema: None,
    };
    assert_eq!(
        render_python_tool_stub(&freeform),
        "async def tools.apply_patch(input: str, /) -> Any: ..."
    );

    let mut mcp = function_tool(
        "mcp__docs__search",
        json!({"type": "object", "properties": {}}),
    );
    mcp.output_schema = Some(json!({
        "type": "object",
        "properties": {
            "content": {"type": "array", "items": {"type": "object"}},
            "isError": {"type": "boolean"},
        },
    }));
    assert_eq!(
        render_python_tool_stub(&mcp),
        "async def tools.mcp__docs__search(**kwargs: Any) -> CallToolResult: ..."
    );
}

#[test]
fn refs_nullable_unions_and_oversized_schemas() {
    let tool = function_tool(
        "lookup",
        json!({
            "type": "object",
            "$defs": {"id": {"type": ["string", "null"]}},
            "properties": {"id": {"$ref": "#/$defs/id"}},
            "required": ["id"],
        }),
    );
    assert_eq!(
        render_python_tool_stub(&tool),
        "async def tools.lookup(\n    *,\n    id: str | None,\n) -> Any: ..."
    );

    let mut small_budget = tool;
    small_budget.input_schema_max_bytes = Some(4);
    assert_eq!(
        render_python_tool_stub(&small_budget),
        "async def tools.lookup(**kwargs: Any) -> Any: ..."
    );
}

#[test]
fn python_exec_description_lists_tools_in_code_mode_only() {
    let tools = vec![function_tool(
        "view_image",
        json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    )];
    let (description, definitions) = build_python_exec_tool_description_and_definitions(
        &tools,
        &BTreeMap::new(),
        /*default_exec_yield_time_ms*/ 5_000,
        /*code_mode_only*/ true,
        ImageDetailVisibility::Hidden,
        DeferredToolDiscovery::Catalog,
        /*incremental_exec_enabled*/ true,
    );
    assert!(description.starts_with("Run Python code"));
    assert!(description.contains("Defaults to 5000 ms."));
    assert!(description.contains(
        "```python\nasync def tools.view_image(\n    *,\n    path: str,\n) -> Any: ...\n```"
    ));
    assert!(!description.contains("```ts"));
    assert_eq!(
        definitions
            .entries
            .iter()
            .map(|entry| entry.key.as_str())
            .collect::<Vec<_>>(),
        vec!["Tool: tools.view_image"]
    );
    assert!(
        definitions
            .helper_signatures
            .contains("store(key: str, value: Any)")
    );

    let (direct_description, _) = build_python_exec_tool_description_and_definitions(
        &tools,
        &BTreeMap::new(),
        5_000,
        /*code_mode_only*/ false,
        ImageDetailVisibility::Hidden,
        DeferredToolDiscovery::RankedSearch,
        false,
    );
    assert!(!direct_description.contains("### `view_image`"));
    assert!(direct_description.contains("tools.tool_search(query="));
}

#[test]
fn python_pragma_is_parsed() {
    let parsed = crate::parse_exec_source_for_language(
        "# @exec: {\"yield_time_ms\": 10}\nprint('hi')",
        CodeModeLanguage::Python,
    )
    .unwrap();
    assert_eq!(parsed.code, "print('hi')");
    assert_eq!(parsed.yield_time_ms, Some(10));

    // A JavaScript pragma is ordinary (invalid) source in Python.
    let parsed = crate::parse_exec_source_for_language(
        "// @exec: {\"yield_time_ms\": 10}\nprint('hi')",
        CodeModeLanguage::Python,
    )
    .unwrap();
    assert_eq!(parsed.yield_time_ms, None);
}
