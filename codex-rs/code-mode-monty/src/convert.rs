//! Conversions between tool JSON and Monty values.

use monty_types::MontyObject;
use monty_types::ObjectRef;
use serde_json::Map;
use serde_json::Number;
use serde_json::Value as JsonValue;

/// Nesting deeper than this is cut off rather than recursed into.
const MAX_DEPTH: usize = 128;

pub(crate) fn json_to_monty(value: &JsonValue) -> MontyObject {
    json_to_monty_at(value, 0)
}

fn json_to_monty_at(value: &JsonValue, depth: usize) -> MontyObject {
    if depth > MAX_DEPTH {
        return MontyObject::none();
    }
    match value {
        JsonValue::Null => MontyObject::none(),
        JsonValue::Bool(value) => MontyObject::bool(*value),
        JsonValue::Number(number) => match number.as_i64() {
            Some(value) => MontyObject::int(value),
            None => MontyObject::float(number.as_f64().unwrap_or(f64::NAN)),
        },
        JsonValue::String(text) => MontyObject::string(text.as_str()),
        JsonValue::Array(items) => {
            MontyObject::list(items.iter().map(|item| json_to_monty_at(item, depth + 1)))
        }
        JsonValue::Object(map) => MontyObject::dict(map.iter().map(|(key, value)| {
            (
                MontyObject::string(key.as_str()),
                json_to_monty_at(value, depth + 1),
            )
        })),
    }
}

/// Converts a sandbox value to JSON. Values JSON cannot represent are an error, so the
/// script sees a `TypeError` instead of a silently mangled argument.
pub(crate) fn monty_to_json(value: ObjectRef<'_>) -> Result<JsonValue, String> {
    monty_to_json_at(value, 0)
}

fn monty_to_json_at(value: ObjectRef<'_>, depth: usize) -> Result<JsonValue, String> {
    if depth > MAX_DEPTH {
        return Err("value is nested too deeply to convert to JSON".to_string());
    }
    if value.type_name() == "NoneType" {
        return Ok(JsonValue::Null);
    }
    if let Some(value) = value.as_bool() {
        return Ok(JsonValue::Bool(value));
    }
    if let Some(text) = value.as_str() {
        return Ok(JsonValue::String(text.to_string()));
    }
    if matches!(value.type_name(), "int") {
        return match value.as_int() {
            Some(int) => Ok(JsonValue::Number(int.into())),
            None => Err("int is too large to convert to JSON".to_string()),
        };
    }
    if matches!(value.type_name(), "float") {
        let float = value.as_float().unwrap_or(f64::NAN);
        return Number::from_f64(float)
            .map(JsonValue::Number)
            .ok_or_else(|| format!("float {float} cannot be converted to JSON"));
    }
    if matches!(value.type_name(), "dict")
        && let Some(pairs) = value.pairs()
    {
        let mut map = Map::with_capacity(pairs.len());
        for (key, item) in pairs {
            let Some(key) = key.as_str() else {
                return Err(format!(
                    "dict keys must be str to convert to JSON, got {}",
                    key.type_name()
                ));
            };
            map.insert(key.to_string(), monty_to_json_at(item, depth + 1)?);
        }
        return Ok(JsonValue::Object(map));
    }
    if let Some(items) = value.items() {
        return items
            .into_iter()
            .map(|item| monty_to_json_at(item, depth + 1))
            .collect::<Result<Vec<_>, _>>()
            .map(JsonValue::Array);
    }
    Err(format!(
        "values of type {} cannot be converted to JSON",
        value.type_name()
    ))
}

/// Text for `text()`, `notify()` and the final expression: strings as-is, other values as
/// JSON when they convert, otherwise their `repr()`.
pub(crate) fn display_text(value: ObjectRef<'_>) -> String {
    if let Some(text) = value.as_str() {
        return text.to_string();
    }
    match monty_to_json(value) {
        Ok(json) => json.to_string(),
        Err(_) => value.py_repr(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn json_round_trips_through_monty() {
        let value = json!({
            "null": null,
            "flag": true,
            "count": 3,
            "ratio": 0.5,
            "name": "codex",
            "items": [1, "two", [3]],
            "nested": {"a": {"b": false}},
        });
        let monty = json_to_monty(&value);
        assert_eq!(monty_to_json(monty.as_ref()).unwrap(), value);
    }

    #[test]
    fn tuples_become_arrays_and_bad_keys_fail() {
        let tuple = MontyObject::tuple([MontyObject::int(1), MontyObject::string("x")]);
        assert_eq!(monty_to_json(tuple.as_ref()).unwrap(), json!([1, "x"]));

        let bad = MontyObject::dict([(MontyObject::int(1), MontyObject::none())]);
        assert_eq!(
            monty_to_json(bad.as_ref()).unwrap_err(),
            "dict keys must be str to convert to JSON, got int"
        );
    }

    #[test]
    fn display_text_prefers_plain_strings() {
        assert_eq!(display_text(MontyObject::string("hi").as_ref()), "hi");
        assert_eq!(
            display_text(json_to_monty(&json!({"a": [1]})).as_ref()),
            r#"{"a":[1]}"#
        );
        assert_eq!(
            display_text(MontyObject::bytes(b"ab".to_vec()).as_ref()),
            "b'ab'"
        );
    }
}
