//! Lightweight types mirroring the Google Gemini API structures needed for token counting.
//!
//! These types support serde serialization/deserialization for JSON interop.
//! They are intentionally minimal — only the fields relevant to token counting
//! are included.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A content message in a conversation, containing a role and parts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Content {
    /// The role of the content author (e.g., "user", "model").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// The parts that make up this content message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<Part>>,
}

/// A single part of a content message.
///
/// Each part contains exactly one of the possible content types:
/// text, function_call, or function_response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Part {
    /// Plain text content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// A function call made by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,

    /// A response to a function call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
}

/// A function call made by the model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FunctionCall {
    /// The name of the function to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// The arguments to pass to the function, as a JSON-like map.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<HashMap<String, serde_json::Value>>,
}

/// A response from a function call.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FunctionResponse {
    /// The name of the function that was called.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// The response data from the function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<HashMap<String, serde_json::Value>>,
}

/// A tool definition containing function declarations.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Tool {
    /// The function declarations that make up this tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_declarations: Option<Vec<FunctionDeclaration>>,
}

/// A declaration of a function that the model can call.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FunctionDeclaration {
    /// The name of the function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// A description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The schema for the function's parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Schema>,

    /// The schema for the function's response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Schema>,
}

/// A JSON Schema definition used to describe function parameters and responses.
///
/// This is a recursive type that can describe nested object structures.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Schema {
    /// The data type (e.g., "STRING", "NUMBER", "OBJECT", "ARRAY").
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub schema_type: Option<String>,

    /// The format of the data (e.g., "int32", "date-time").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// A description of what this schema represents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The title of this schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// A default value for this schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,

    /// Allowed enum values for this field.
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,

    /// Required property names (for object types).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,

    /// Property ordering hints.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub property_ordering: Option<Vec<String>>,

    /// Schema for array items.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<Schema>>,

    /// Named properties (for object types).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Schema>>,

    /// An example value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub example: Option<serde_json::Value>,
}

/// Input contents, accepting either a text string or structured Content objects.
///
/// Matches the Python SDK's flexible content input where `count_tokens` and
/// `compute_tokens` accept both plain strings and structured Content objects.
///
/// # From implementations
///
/// - `&str` → wraps as a single user Content with one text Part
/// - `&[Content]` → uses the Content slice directly
/// - `&Vec<Content>` → delegates to the slice implementation
pub enum Contents<'a> {
    /// Plain text input (will be wrapped as a user Content).
    Text(&'a str),
    /// Structured Content objects.
    Structured(&'a [Content]),
}

impl<'a> From<&'a str> for Contents<'a> {
    fn from(s: &'a str) -> Self {
        Contents::Text(s)
    }
}

impl<'a> From<&'a [Content]> for Contents<'a> {
    fn from(c: &'a [Content]) -> Self {
        Contents::Structured(c)
    }
}

impl<'a> From<&'a Vec<Content>> for Contents<'a> {
    fn from(c: &'a Vec<Content>) -> Self {
        Contents::Structured(c.as_slice())
    }
}

/// Result of counting tokens, matching the Python SDK's `CountTokensResult`.
#[derive(Debug, Clone)]
pub struct CountTokensResult {
    /// The total number of tokens.
    pub total_tokens: usize,
}

impl std::fmt::Display for CountTokensResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "total_tokens={}", self.total_tokens)
    }
}

/// Information about tokens for a single content part,
/// matching the Python SDK's `TokensInfo`.
#[derive(Debug, Clone)]
pub struct TokensInfo {
    /// The token IDs in the vocabulary.
    pub token_ids: Vec<u32>,
    /// The token pieces as byte sequences (UTF-8 encoded, with SentencePiece
    /// space markers replaced by actual spaces).
    pub tokens: Vec<Vec<u8>>,
    /// The role of the content this part belongs to (e.g., "user", "model").
    pub role: Option<String>,
}

/// Result of computing tokens, matching the Python SDK's `ComputeTokensResult`.
#[derive(Debug, Clone)]
pub struct ComputeTokensResult {
    /// Token information for each content part.
    pub tokens_info: Vec<TokensInfo>,
}

/// Configuration for `count_tokens`, matching the Python SDK's `CountTokensConfig`.
///
/// Provides optional tools, system instruction, and response schema that
/// contribute additional tokens to the count.
#[derive(Debug, Clone, Default)]
pub struct CountTokensConfig {
    /// Tool definitions whose declarations contribute tokens.
    pub tools: Option<Vec<Tool>>,
    /// System instruction content that contributes tokens.
    pub system_instruction: Option<Content>,
    /// Response schema that contributes tokens.
    pub response_schema: Option<Schema>,
}

// <FILE>src/types.rs</FILE> - <DESC>Lightweight Gemini content types for token counting</DESC>
// <VERS>END OF VERSION: 0.2.0</VERS>
