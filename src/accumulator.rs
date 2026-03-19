//! Text accumulation logic ported from the official Google Python SDK.
//!
//! The [`TextAccumulator`] traverses structured [`Content`], [`Tool`], and [`Schema`]
//! objects, extracting all text segments that should be counted as tokens.
//! This matches the behavior of `_TextsAccumulator` in
//! `google/genai/local_tokenizer.py`.

use crate::types::*;

/// Accumulates countable text strings from structured Gemini API objects.
///
/// This is a faithful port of the Python SDK's `_TextsAccumulator` class.
/// It traverses `Content`, `Tool`, `FunctionCall`, `FunctionResponse`, and
/// `Schema` objects, extracting all text that Google counts when computing
/// token counts.
///
/// # How text is extracted
///
/// - **Text parts**: the text string itself
/// - **Function calls**: the function name, plus all dict keys and string values
///   from the args
/// - **Function responses**: the function name, plus all dict keys and string
///   values from the response
/// - **Function declarations** (tools): the name, description, plus recursive
///   schema traversal of parameters and response
/// - **Schemas**: format, description, enum values, required field names,
///   property keys, and recursive traversal of nested schemas
pub struct TextAccumulator {
    texts: Vec<String>,
}

impl TextAccumulator {
    /// Creates a new empty accumulator.
    pub fn new() -> Self {
        Self { texts: Vec::new() }
    }

    /// Returns the accumulated text segments.
    pub fn get_texts(&self) -> &[String] {
        &self.texts
    }

    /// Consumes the accumulator and returns the accumulated text segments.
    pub fn into_texts(self) -> Vec<String> {
        self.texts
    }

    /// Adds all text from multiple content objects.
    pub fn add_contents(&mut self, contents: &[Content]) {
        for content in contents {
            self.add_content(content);
        }
    }

    /// Adds all countable text from a single content object.
    ///
    /// Processes each part in the content by delegating to [`add_part`](Self::add_part).
    pub fn add_content(&mut self, content: &Content) {
        if let Some(parts) = &content.parts {
            for part in parts {
                self.add_part(part);
            }
        }
    }

    /// Adds countable text from a single content part.
    ///
    /// Processes the part's fields:
    /// - Text: appends the text directly
    /// - Function calls: delegates to [`add_function_call`](Self::add_function_call)
    /// - Function responses: delegates to [`add_function_response`](Self::add_function_response)
    pub fn add_part(&mut self, part: &Part) {
        if let Some(fc) = &part.function_call {
            self.add_function_call(fc);
        }
        if let Some(fr) = &part.function_response {
            self.add_function_response(fr);
        }
        if let Some(text) = &part.text {
            self.texts.push(text.clone());
        }
    }

    /// Adds countable text from a function call.
    ///
    /// Extracts the function name and traverses the args dictionary,
    /// collecting all keys and string values.
    pub fn add_function_call(&mut self, function_call: &FunctionCall) {
        if let Some(name) = &function_call.name {
            self.texts.push(name.clone());
        }
        if let Some(args) = &function_call.args {
            self.dict_traverse(args);
        }
    }

    /// Adds countable text from multiple tools.
    pub fn add_tools(&mut self, tools: &[Tool]) {
        for tool in tools {
            self.add_tool(tool);
        }
    }

    /// Adds countable text from a single tool definition.
    ///
    /// Processes each function declaration in the tool.
    pub fn add_tool(&mut self, tool: &Tool) {
        if let Some(declarations) = &tool.function_declarations {
            for decl in declarations {
                self.add_function_declaration(decl);
            }
        }
    }

    /// Adds countable text from multiple function responses.
    pub fn add_function_responses(&mut self, responses: &[FunctionResponse]) {
        for response in responses {
            self.add_function_response(response);
        }
    }

    /// Adds countable text from a function response.
    ///
    /// Extracts the function name and traverses the response dictionary,
    /// collecting all keys and string values.
    pub fn add_function_response(&mut self, function_response: &FunctionResponse) {
        if let Some(name) = &function_response.name {
            self.texts.push(name.clone());
        }
        if let Some(response) = &function_response.response {
            self.dict_traverse(response);
        }
    }

    /// Adds countable text from a function declaration.
    ///
    /// Extracts the name, description, and recursively processes the
    /// parameter and response schemas.
    fn add_function_declaration(&mut self, decl: &FunctionDeclaration) {
        if let Some(name) = &decl.name {
            self.texts.push(name.clone());
        }
        if let Some(description) = &decl.description {
            self.texts.push(description.clone());
        }
        if let Some(parameters) = &decl.parameters {
            self.add_schema(parameters);
        }
        if let Some(response) = &decl.response {
            self.add_schema(response);
        }
    }

    /// Adds countable text from a schema definition.
    ///
    /// Extracts format, description, enum values, required field names,
    /// property keys, and recursively processes nested schemas (items,
    /// properties, examples).
    pub fn add_schema(&mut self, schema: &Schema) {
        // Note: schema.type and schema.title are tracked but NOT added to texts,
        // matching the Python SDK behavior.
        if let Some(format) = &schema.format {
            self.texts.push(format.clone());
        }
        if let Some(description) = &schema.description {
            self.texts.push(description.clone());
        }
        if let Some(enum_values) = &schema.enum_values {
            for v in enum_values {
                self.texts.push(v.clone());
            }
        }
        if let Some(required) = &schema.required {
            for r in required {
                self.texts.push(r.clone());
            }
        }
        if let Some(items) = &schema.items {
            self.add_schema(items);
        }
        if let Some(properties) = &schema.properties {
            for (key, value) in properties {
                self.texts.push(key.clone());
                self.add_schema(value);
            }
        }
        if let Some(example) = &schema.example {
            self.any_traverse(example);
        }
    }

    /// Traverses a dictionary (JSON object), adding all keys and recursively
    /// processing all values.
    fn dict_traverse(&mut self, d: &std::collections::HashMap<String, serde_json::Value>) {
        // Add all keys
        let keys: Vec<String> = d.keys().cloned().collect();
        self.texts.extend(keys);

        // Traverse all values
        for val in d.values() {
            self.any_traverse(val);
        }
    }

    /// Traverses an arbitrary JSON value, adding strings and recursing into
    /// objects and arrays.
    fn any_traverse(&mut self, value: &serde_json::Value) {
        match value {
            serde_json::Value::String(s) => {
                self.texts.push(s.clone());
            }
            serde_json::Value::Object(map) => {
                // Collect keys
                let keys: Vec<String> = map.keys().cloned().collect();
                self.texts.extend(keys);
                // Recurse into values
                for val in map.values() {
                    self.any_traverse(val);
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.any_traverse(item);
                }
            }
            // Numbers, bools, nulls are not added to texts (matches Python SDK)
            _ => {}
        }
    }
}

impl Default for TextAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_empty_accumulator() {
        let acc = TextAccumulator::new();
        assert!(acc.get_texts().is_empty());
    }

    #[test]
    fn test_add_text_content() {
        let mut acc = TextAccumulator::new();
        let content = Content {
            role: Some("user".to_string()),
            parts: Some(vec![Part {
                text: Some("Hello, world!".to_string()),
                ..Default::default()
            }]),
        };
        acc.add_content(&content);
        assert_eq!(acc.get_texts(), &["Hello, world!"]);
    }

    #[test]
    fn test_add_function_call() {
        let mut acc = TextAccumulator::new();
        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            serde_json::Value::String("weather".to_string()),
        );
        args.insert(
            "location".to_string(),
            serde_json::Value::String("NYC".to_string()),
        );

        let fc = FunctionCall {
            name: Some("search".to_string()),
            args: Some(args),
        };
        acc.add_function_call(&fc);

        let texts = acc.get_texts();
        assert!(texts.contains(&"search".to_string()));
        assert!(texts.contains(&"query".to_string()));
        assert!(texts.contains(&"location".to_string()));
        assert!(texts.contains(&"weather".to_string()));
        assert!(texts.contains(&"NYC".to_string()));
    }

    #[test]
    fn test_add_function_response() {
        let mut acc = TextAccumulator::new();
        let mut response = HashMap::new();
        response.insert(
            "result".to_string(),
            serde_json::Value::String("sunny".to_string()),
        );

        let fr = FunctionResponse {
            name: Some("search".to_string()),
            response: Some(response),
        };
        acc.add_function_response(&fr);

        let texts = acc.get_texts();
        assert!(texts.contains(&"search".to_string()));
        assert!(texts.contains(&"result".to_string()));
        assert!(texts.contains(&"sunny".to_string()));
    }

    #[test]
    fn test_add_schema_with_properties() {
        let mut acc = TextAccumulator::new();
        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            Schema {
                schema_type: Some("STRING".to_string()),
                description: Some("The user's name".to_string()),
                ..Default::default()
            },
        );

        let schema = Schema {
            schema_type: Some("OBJECT".to_string()),
            description: Some("A user object".to_string()),
            required: Some(vec!["name".to_string()]),
            properties: Some(properties),
            ..Default::default()
        };
        acc.add_schema(&schema);

        let texts = acc.get_texts();
        assert!(texts.contains(&"A user object".to_string()));
        assert!(texts.contains(&"name".to_string()));
        // Property key "name" and required "name" both added
        assert!(texts.contains(&"The user's name".to_string()));
    }

    #[test]
    fn test_add_tool() {
        let mut acc = TextAccumulator::new();
        let tool = Tool {
            function_declarations: Some(vec![FunctionDeclaration {
                name: Some("get_weather".to_string()),
                description: Some("Gets the weather for a location".to_string()),
                parameters: Some(Schema {
                    schema_type: Some("OBJECT".to_string()),
                    properties: Some({
                        let mut props = HashMap::new();
                        props.insert(
                            "location".to_string(),
                            Schema {
                                schema_type: Some("STRING".to_string()),
                                description: Some("The city name".to_string()),
                                ..Default::default()
                            },
                        );
                        props
                    }),
                    required: Some(vec!["location".to_string()]),
                    ..Default::default()
                }),
                response: None,
            }]),
        };
        acc.add_tool(&tool);

        let texts = acc.get_texts();
        assert!(texts.contains(&"get_weather".to_string()));
        assert!(texts.contains(&"Gets the weather for a location".to_string()));
        assert!(texts.contains(&"location".to_string())); // property key
        assert!(texts.contains(&"The city name".to_string()));
    }

    #[test]
    fn test_schema_enum_values() {
        let mut acc = TextAccumulator::new();
        let schema = Schema {
            schema_type: Some("STRING".to_string()),
            enum_values: Some(vec![
                "red".to_string(),
                "green".to_string(),
                "blue".to_string(),
            ]),
            ..Default::default()
        };
        acc.add_schema(&schema);

        let texts = acc.get_texts();
        assert!(texts.contains(&"red".to_string()));
        assert!(texts.contains(&"green".to_string()));
        assert!(texts.contains(&"blue".to_string()));
    }

    #[test]
    fn test_any_traverse_nested() {
        let mut acc = TextAccumulator::new();
        let mut args = HashMap::new();
        args.insert(
            "data".to_string(),
            serde_json::json!({"nested_key": "nested_value", "list": ["a", "b"]}),
        );
        let fc = FunctionCall {
            name: Some("test_fn".to_string()),
            args: Some(args),
        };
        acc.add_function_call(&fc);

        let texts = acc.get_texts();
        assert!(texts.contains(&"test_fn".to_string()));
        assert!(texts.contains(&"data".to_string()));
        assert!(texts.contains(&"nested_key".to_string()));
        assert!(texts.contains(&"nested_value".to_string()));
        assert!(texts.contains(&"list".to_string()));
        assert!(texts.contains(&"a".to_string()));
        assert!(texts.contains(&"b".to_string()));
    }

    #[test]
    fn test_content_with_function_call_part() {
        let mut acc = TextAccumulator::new();
        let mut args = HashMap::new();
        args.insert(
            "q".to_string(),
            serde_json::Value::String("test".to_string()),
        );
        let content = Content {
            role: Some("model".to_string()),
            parts: Some(vec![Part {
                function_call: Some(FunctionCall {
                    name: Some("search".to_string()),
                    args: Some(args),
                }),
                ..Default::default()
            }]),
        };
        acc.add_content(&content);

        let texts = acc.get_texts();
        assert!(texts.contains(&"search".to_string()));
        assert!(texts.contains(&"q".to_string()));
        assert!(texts.contains(&"test".to_string()));
    }
}

// <FILE>src/accumulator.rs</FILE> - <DESC>Port of Google Python SDK's _TextsAccumulator for extracting countable text</DESC>
// <VERS>END OF VERSION: 0.2.0</VERS>
