//! # gemini-tokenizer
//!
//! Authoritative Gemini tokenizer for Rust, ported from the official
//! [Google Python SDK](https://github.com/googleapis/python-genai) (v1.6.20).
//!
//! All Gemini models (gemini-2.0-flash, gemini-2.5-pro, gemini-3-pro-preview, etc.)
//! use the same tokenizer: the Gemma 3 SentencePiece model with a vocabulary of
//! 262,144 tokens. This crate embeds that model and provides a fast, local
//! tokenizer that matches the official Google Python SDK's behavior.
//!
//! ## Quick Start
//!
//! ```rust
//! use gemini_tokenizer::LocalTokenizer;
//!
//! let tokenizer = LocalTokenizer::new("gemini-2.5-pro").expect("failed to load tokenizer");
//!
//! // Count tokens in plain text
//! let result = tokenizer.count_tokens("What is your name?", None);
//! assert_eq!(result.total_tokens, 5);
//!
//! // Get individual token details
//! let result = tokenizer.compute_tokens("Hello, world!");
//! for info in &result.tokens_info {
//!     for (id, token) in info.token_ids.iter().zip(&info.tokens) {
//!         println!("id={}, token={:?}", id, token);
//!     }
//! }
//! ```
//!
//! ## Structured Content
//!
//! The tokenizer also counts tokens in structured Gemini API content objects,
//! matching the Google Python SDK's `_TextsAccumulator` logic:
//!
//! ```rust
//! use gemini_tokenizer::{LocalTokenizer, Content, Part, CountTokensConfig, Tool,
//!     FunctionDeclaration, Schema};
//!
//! let tokenizer = LocalTokenizer::new("gemini-2.5-pro").expect("failed to load tokenizer");
//!
//! let contents = vec![Content {
//!     role: Some("user".to_string()),
//!     parts: Some(vec![Part {
//!         text: Some("What is the weather in NYC?".to_string()),
//!         ..Default::default()
//!     }]),
//! }];
//!
//! let result = tokenizer.count_tokens(contents.as_slice(), None);
//! assert!(result.total_tokens > 0);
//! ```

pub mod accumulator;
pub mod types;

pub use accumulator::TextAccumulator;
pub use types::*;

use sentencepiece::SentencePieceProcessor;
use std::sync::{Arc, OnceLock};

/// The expected SHA-256 hash of the embedded SentencePiece model.
pub const MODEL_SHA256: &str = "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c";

/// The expected vocabulary size of the Gemma 3 tokenizer.
pub const VOCAB_SIZE: usize = 262_144;

/// The embedded SentencePiece model bytes (Gemma 3, 262k vocab).
///
/// This is the same model used by all Gemini models (2.0, 2.5, 3.0).
/// Source: <https://github.com/google/gemma_pytorch>
static MODEL_BYTES: &[u8] = include_bytes!("../resources/gemma3_cleaned_262144_v2.spiece.model");

static GLOBAL_PROCESSOR: OnceLock<Arc<SentencePieceProcessor>> = OnceLock::new();

/// Supported model names, matching the Python SDK's model-to-tokenizer mapping.
///
/// Source: `google/genai/_local_tokenizer_loader.py` in python-genai v1.6.20.
/// All models map to the "gemma3" tokenizer.
const SUPPORTED_MODELS: &[&str] = &[
    // Dynamic model aliases
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    // Stable versioned models
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-exp-03-25",
    "gemini-live-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gemini-3-pro-preview",
];

/// Errors that can occur when creating or using the tokenizer.
#[derive(Debug)]
pub enum TokenizerError {
    /// The SentencePiece model failed to load.
    ModelLoadError(String),

    /// The embedded model's hash does not match the expected value.
    HashMismatch { expected: String, actual: String },

    /// The requested model name is not supported.
    UnsupportedModel(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::ModelLoadError(msg) => {
                write!(f, "failed to load SentencePiece model: {}", msg)
            }
            TokenizerError::HashMismatch { expected, actual } => {
                write!(
                    f,
                    "model hash mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            TokenizerError::UnsupportedModel(name) => {
                write!(
                    f,
                    "model {} is not supported. Supported models: {}",
                    name,
                    SUPPORTED_MODELS.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

/// The local Gemini tokenizer.
///
/// Matches the Python SDK's `LocalTokenizer` interface. Wraps a SentencePiece
/// processor loaded with the Gemma 3 model used by all Gemini models. The model
/// is embedded in the binary at compile time.
///
/// # Example
///
/// ```
/// use gemini_tokenizer::LocalTokenizer;
///
/// let tok = LocalTokenizer::new("gemini-2.5-pro").unwrap();
/// let result = tok.count_tokens("Hello, world!", None);
/// println!("{}", result); // total_tokens=4
/// ```
#[derive(Debug)]
pub struct LocalTokenizer {
    processor: Arc<SentencePieceProcessor>,
    model_name: String,
}

impl LocalTokenizer {
    /// Creates a new tokenizer for the given Gemini model.
    ///
    /// Validates the model name against the supported list (matching the Python
    /// SDK's `_local_tokenizer_loader.py`) and loads the embedded SentencePiece
    /// model.
    ///
    /// # Errors
    ///
    /// - [`TokenizerError::UnsupportedModel`] if the model name is not recognized.
    /// - [`TokenizerError::ModelLoadError`] if the SentencePiece model fails to
    ///   deserialize.
    pub fn new(model_name: &str) -> Result<Self, TokenizerError> {
        if !SUPPORTED_MODELS.contains(&model_name) {
            return Err(TokenizerError::UnsupportedModel(model_name.to_string()));
        }
        let processor = GLOBAL_PROCESSOR
            .get_or_init(|| {
                let p = SentencePieceProcessor::from_serialized_proto(MODEL_BYTES)
                    .expect("Critical: Embedded tokenizer model is corrupt");
                Arc::new(p)
            })
            .clone();

        Ok(Self {
            processor,
            model_name: model_name.to_string(),
        })
    }

    /// Returns the model name this tokenizer was created for.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Returns the vocabulary size of the loaded model.
    pub fn vocab_size(&self) -> usize {
        self.processor.len()
    }

    /// Counts the number of tokens in the given contents.
    ///
    /// Accepts either a plain text string or structured Content objects via the
    /// [`Contents`] enum. An optional [`CountTokensConfig`] can provide tools,
    /// system instruction, and response schema that contribute additional tokens.
    ///
    /// This matches the Python SDK's `LocalTokenizer.count_tokens()` method.
    ///
    /// # Example
    ///
    /// ```
    /// use gemini_tokenizer::LocalTokenizer;
    ///
    /// let tok = LocalTokenizer::new("gemini-2.0-flash").unwrap();
    ///
    /// // Plain text
    /// let result = tok.count_tokens("What is your name?", None);
    /// assert_eq!(result.total_tokens, 5);
    /// ```
    pub fn count_tokens<'a>(
        &self,
        contents: impl Into<Contents<'a>>,
        config: Option<&CountTokensConfig>,
    ) -> CountTokensResult {
        let content_vec = contents_to_vec(contents.into());
        let mut acc = TextAccumulator::new();
        acc.add_contents(&content_vec);

        if let Some(config) = config {
            if let Some(tools) = &config.tools {
                acc.add_tools(tools);
            }
            if let Some(schema) = &config.response_schema {
                acc.add_schema(schema);
            }
            if let Some(system_instruction) = &config.system_instruction {
                acc.add_content(system_instruction);
            }
        }

        let mut total = 0;
        for text in acc.get_texts() {
            total += match self.processor.encode(text) {
                Ok(pieces) => pieces.len(),
                Err(_) => 0,
            };
        }

        CountTokensResult {
            total_tokens: total,
        }
    }

    /// Computes token IDs and byte pieces for the given contents.
    ///
    /// Returns a [`ComputeTokensResult`] with one [`TokensInfo`] entry per
    /// content part, preserving the role from the parent Content object.
    ///
    /// This matches the Python SDK's `LocalTokenizer.compute_tokens()` method.
    ///
    /// # Example
    ///
    /// ```
    /// use gemini_tokenizer::LocalTokenizer;
    ///
    /// let tok = LocalTokenizer::new("gemini-2.5-pro").unwrap();
    /// let result = tok.compute_tokens("Hello");
    /// assert_eq!(result.tokens_info.len(), 1);
    /// assert!(!result.tokens_info[0].token_ids.is_empty());
    /// assert_eq!(result.tokens_info[0].role, Some("user".to_string()));
    /// ```
    pub fn compute_tokens<'a>(&self, contents: impl Into<Contents<'a>>) -> ComputeTokensResult {
        let content_vec = contents_to_vec(contents.into());
        let mut tokens_info = Vec::new();

        for content in &content_vec {
            if let Some(parts) = &content.parts {
                for part in parts {
                    let mut acc = TextAccumulator::new();
                    acc.add_part(part);

                    let mut all_ids = Vec::new();
                    let mut all_tokens = Vec::new();
                    for text in acc.get_texts() {
                        if let Ok(pieces) = self.processor.encode(text) {
                            for p in pieces {
                                all_ids.push(p.id);
                                all_tokens.push(token_piece_to_bytes(&p.piece));
                            }
                        }
                    }

                    tokens_info.push(TokensInfo {
                        token_ids: all_ids,
                        tokens: all_tokens,
                        role: content.role.clone(),
                    });
                }
            }
        }

        ComputeTokensResult { tokens_info }
    }

    /// Returns a reference to the underlying SentencePiece processor.
    pub fn processor(&self) -> &SentencePieceProcessor {
        &self.processor
    }
}

/// Converts a [`Contents`] input to an owned `Vec<Content>`.
///
/// For text input, wraps the string as a single user Content with one text Part,
/// matching the Python SDK's `t.t_contents()` behavior for string input.
fn contents_to_vec(contents: Contents<'_>) -> Vec<Content> {
    match contents {
        Contents::Text(s) => vec![Content {
            role: Some("user".to_string()),
            parts: Some(vec![Part {
                text: Some(s.to_string()),
                ..Default::default()
            }]),
        }],
        Contents::Structured(c) => c.to_vec(),
    }
}

/// Converts a SentencePiece token piece string to bytes.
///
/// Matches the Python SDK's `_token_str_to_bytes`:
/// - Byte-fallback tokens (`<0xXX>`) → single byte
/// - Normal tokens → replace `▁` with space, encode as UTF-8
fn token_piece_to_bytes(piece: &str) -> Vec<u8> {
    if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
        if let Ok(val) = u8::from_str_radix(&piece[3..5], 16) {
            return vec![val];
        }
    }
    piece.replace('\u{2581}', " ").into_bytes()
}

/// Verifies that the embedded model's SHA-256 hash matches the expected value.
///
/// Useful in tests and CI to ensure the embedded model has not been corrupted.
pub fn verify_model_hash() -> Result<(), TokenizerError> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(MODEL_BYTES);
    let actual = format!("{:x}", hasher.finalize());
    if actual == MODEL_SHA256 {
        Ok(())
    } else {
        Err(TokenizerError::HashMismatch {
            expected: MODEL_SHA256.to_string(),
            actual,
        })
    }
}

/// Returns the list of supported Gemini model names.
pub fn supported_models() -> &'static [&'static str] {
    SUPPORTED_MODELS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_verify_embedded_model_hash() {
        verify_model_hash().expect("embedded model hash should match");
    }

    #[test]
    fn test_vocab_size() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");
        assert_eq!(tok.vocab_size(), VOCAB_SIZE);
    }

    #[test]
    fn test_model_name() {
        let tok = LocalTokenizer::new("gemini-2.0-flash").expect("tokenizer should load");
        assert_eq!(tok.model_name(), "gemini-2.0-flash");
    }

    #[test]
    fn test_unsupported_model() {
        let err = LocalTokenizer::new("gpt-4").unwrap_err();
        match err {
            TokenizerError::UnsupportedModel(name) => assert_eq!(name, "gpt-4"),
            _ => panic!("expected UnsupportedModel error"),
        }
    }

    #[test]
    fn test_all_supported_models() {
        for model in SUPPORTED_MODELS {
            LocalTokenizer::new(model).unwrap_or_else(|_| panic!("{} should be supported", model));
        }
    }

    #[test]
    fn test_count_tokens_text() {
        let tok = LocalTokenizer::new("gemini-2.0-flash-001").expect("tokenizer should load");
        let result = tok.count_tokens("What is your name?", None);
        assert_eq!(result.total_tokens, 5);
    }

    #[test]
    fn test_count_tokens_empty() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");
        let result = tok.count_tokens("", None);
        assert_eq!(result.total_tokens, 0);
    }

    #[test]
    fn test_count_tokens_content() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");
        let contents = vec![Content {
            role: Some("user".to_string()),
            parts: Some(vec![Part {
                text: Some("Hello, world!".to_string()),
                ..Default::default()
            }]),
        }];
        let result = tok.count_tokens(contents.as_slice(), None);
        let direct = tok.count_tokens("Hello, world!", None);
        assert_eq!(result.total_tokens, direct.total_tokens);
    }

    #[test]
    fn test_count_tokens_vec_ref() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");
        let contents = vec![Content {
            role: Some("user".to_string()),
            parts: Some(vec![Part {
                text: Some("Hello".to_string()),
                ..Default::default()
            }]),
        }];
        // Test that &Vec<Content> works via From impl
        let result = tok.count_tokens(&contents, None);
        assert!(result.total_tokens > 0);
    }

    #[test]
    fn test_count_tokens_function_call() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");

        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            serde_json::Value::String("weather".to_string()),
        );

        let contents = vec![Content {
            role: Some("model".to_string()),
            parts: Some(vec![Part {
                function_call: Some(FunctionCall {
                    name: Some("search".to_string()),
                    args: Some(args),
                }),
                ..Default::default()
            }]),
        }];

        let result = tok.count_tokens(contents.as_slice(), None);
        let expected = tok.count_tokens("search", None).total_tokens
            + tok.count_tokens("query", None).total_tokens
            + tok.count_tokens("weather", None).total_tokens;
        assert_eq!(result.total_tokens, expected);
    }

    #[test]
    fn test_count_tokens_function_response() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");

        let mut response = HashMap::new();
        response.insert(
            "result".to_string(),
            serde_json::Value::String("sunny".to_string()),
        );

        let contents = vec![Content {
            role: Some("model".to_string()),
            parts: Some(vec![Part {
                function_response: Some(FunctionResponse {
                    name: Some("search".to_string()),
                    response: Some(response),
                }),
                ..Default::default()
            }]),
        }];

        let result = tok.count_tokens(contents.as_slice(), None);
        let expected = tok.count_tokens("search", None).total_tokens
            + tok.count_tokens("result", None).total_tokens
            + tok.count_tokens("sunny", None).total_tokens;
        assert_eq!(result.total_tokens, expected);
    }

    #[test]
    fn test_count_tokens_with_tools() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");

        let contents = vec![Content {
            role: Some("user".to_string()),
            parts: Some(vec![Part {
                text: Some("What is the weather?".to_string()),
                ..Default::default()
            }]),
        }];

        let config = CountTokensConfig {
            tools: Some(vec![Tool {
                function_declarations: Some(vec![FunctionDeclaration {
                    name: Some("get_weather".to_string()),
                    description: Some("Gets the current weather".to_string()),
                    parameters: Some(Schema {
                        schema_type: Some("OBJECT".to_string()),
                        properties: Some({
                            let mut props = HashMap::new();
                            props.insert(
                                "city".to_string(),
                                Schema {
                                    schema_type: Some("STRING".to_string()),
                                    description: Some("The city name".to_string()),
                                    ..Default::default()
                                },
                            );
                            props
                        }),
                        required: Some(vec!["city".to_string()]),
                        ..Default::default()
                    }),
                    response: None,
                }]),
            }]),
            ..Default::default()
        };

        let with_tools = tok.count_tokens(contents.as_slice(), Some(&config));
        let without_tools = tok.count_tokens(contents.as_slice(), None);
        assert!(with_tools.total_tokens > without_tools.total_tokens);
    }

    #[test]
    fn test_count_tokens_with_system_instruction() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");

        let contents = vec![Content {
            role: Some("user".to_string()),
            parts: Some(vec![Part {
                text: Some("Hello".to_string()),
                ..Default::default()
            }]),
        }];

        let config = CountTokensConfig {
            system_instruction: Some(Content {
                role: Some("system".to_string()),
                parts: Some(vec![Part {
                    text: Some("You are a helpful assistant.".to_string()),
                    ..Default::default()
                }]),
            }),
            ..Default::default()
        };

        let with_system = tok.count_tokens(contents.as_slice(), Some(&config));
        let without_system = tok.count_tokens(contents.as_slice(), None);
        assert!(with_system.total_tokens > without_system.total_tokens);
    }

    #[test]
    fn test_count_tokens_multiple_parts() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");
        let contents = vec![Content {
            role: Some("user".to_string()),
            parts: Some(vec![
                Part {
                    text: Some("Hello".to_string()),
                    ..Default::default()
                },
                Part {
                    text: Some("World".to_string()),
                    ..Default::default()
                },
            ]),
        }];

        let result = tok.count_tokens(contents.as_slice(), None);
        let expected = tok.count_tokens("Hello", None).total_tokens
            + tok.count_tokens("World", None).total_tokens;
        assert_eq!(result.total_tokens, expected);
    }

    #[test]
    fn test_compute_tokens_text() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");
        let result = tok.compute_tokens("Hello");
        assert_eq!(result.tokens_info.len(), 1);
        assert!(!result.tokens_info[0].token_ids.is_empty());
        assert_eq!(
            result.tokens_info[0].token_ids.len(),
            result.tokens_info[0].tokens.len()
        );
        assert_eq!(result.tokens_info[0].role, Some("user".to_string()));
    }

    #[test]
    fn test_compute_tokens_matches_count() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");
        let text = "The quick brown fox jumps over the lazy dog.";
        let count_result = tok.count_tokens(text, None);
        let compute_result = tok.compute_tokens(text);
        let total_ids: usize = compute_result
            .tokens_info
            .iter()
            .map(|ti| ti.token_ids.len())
            .sum();
        assert_eq!(total_ids, count_result.total_tokens);
    }

    #[test]
    fn test_compute_tokens_preserves_role() {
        let tok = LocalTokenizer::new("gemini-2.5-pro").expect("tokenizer should load");
        let contents = vec![
            Content {
                role: Some("user".to_string()),
                parts: Some(vec![Part {
                    text: Some("Hello".to_string()),
                    ..Default::default()
                }]),
            },
            Content {
                role: Some("model".to_string()),
                parts: Some(vec![Part {
                    text: Some("Hi there".to_string()),
                    ..Default::default()
                }]),
            },
        ];
        let result = tok.compute_tokens(contents.as_slice());
        assert_eq!(result.tokens_info.len(), 2);
        assert_eq!(result.tokens_info[0].role, Some("user".to_string()));
        assert_eq!(result.tokens_info[1].role, Some("model".to_string()));
    }

    #[test]
    fn test_count_tokens_display() {
        let result = CountTokensResult { total_tokens: 42 };
        assert_eq!(format!("{}", result), "total_tokens=42");
    }

    #[test]
    fn test_tokenizer_error_display() {
        let err = TokenizerError::ModelLoadError("test error".to_string());
        assert!(format!("{}", err).contains("test error"));

        let err = TokenizerError::HashMismatch {
            expected: "aaa".to_string(),
            actual: "bbb".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("aaa"));
        assert!(msg.contains("bbb"));

        let err = TokenizerError::UnsupportedModel("gpt-4".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("gpt-4"));
        assert!(msg.contains("not supported"));
    }

    #[test]
    fn test_token_piece_to_bytes_normal() {
        let bytes = token_piece_to_bytes("\u{2581}Hello");
        assert_eq!(bytes, b" Hello");
    }

    #[test]
    fn test_token_piece_to_bytes_hex() {
        let bytes = token_piece_to_bytes("<0xFF>");
        assert_eq!(bytes, vec![0xFF]);
    }

    #[test]
    fn test_supported_models_list() {
        let models = supported_models();
        assert!(models.contains(&"gemini-2.5-pro"));
        assert!(models.contains(&"gemini-3-pro-preview"));
    }
}

// <FILE>src/lib.rs</FILE> - <DESC>Authoritative Gemini tokenizer for Rust, based on the official Google Python SDK</DESC>
// <VERS>END OF VERSION: 0.2.0</VERS>
