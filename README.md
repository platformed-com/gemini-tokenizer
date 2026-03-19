# gemini-tokenizer

## Platformed Note

This is a fork from Crates.io - to get the source tarball you need to do the following:
```
curl https://crates.io/api/v1/crates/gemini-tokenizer/0.2.0/download
```

Community-maintained Gemini tokenizer for Rust, ported from the official
[Google Python GenAI SDK](https://github.com/googleapis/python-genai) (v1.6.20).

> **Disclaimer**: This is an unofficial community port. It is not maintained or supported by Google.

## Overview

All Gemini models — including gemini-2.0-flash, gemini-2.5-pro, gemini-2.5-flash,
gemini-3-pro-preview, and others — use the **same** tokenizer: the Gemma 3
SentencePiece model with a vocabulary of **262,144** tokens.

This crate embeds that model directly in the binary (via `include_bytes!`) and
provides a fast, local tokenizer that produces identical token counts to the
official Google Python SDK. No network access or external files are needed at
runtime.

## Features

- **Python SDK parity** — API mirrors the Python SDK's `LocalTokenizer` interface
  for familiarity.
- **Embedded model** — The SentencePiece model is compiled into your binary. No
  runtime downloads.
- **Faithful port** — Token counting logic is a direct port of the Python SDK's
  `_TextsAccumulator` class from `local_tokenizer.py`.
- **Structured content** — Count tokens in function calls, function responses,
  tool declarations, and schemas, not just plain text.
- **Minimal dependencies** — Only `sentencepiece`, `serde`, `serde_json`, and `sha2`.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
gemini-tokenizer = "0.2"
```

### Count tokens in text

```rust
use gemini_tokenizer::LocalTokenizer;

let tokenizer = LocalTokenizer::new("gemini-2.5-pro")
    .expect("failed to load tokenizer");

let result = tokenizer.count_tokens("What is your name?", None);
assert_eq!(result.total_tokens, 5);
println!("{}", result); // total_tokens=5
```

### Compute individual tokens

```rust
use gemini_tokenizer::LocalTokenizer;

let tokenizer = LocalTokenizer::new("gemini-2.5-pro")
    .expect("failed to load tokenizer");

let result = tokenizer.compute_tokens("Hello, world!");
for info in &result.tokens_info {
    for (id, token) in info.token_ids.iter().zip(&info.tokens) {
        println!("id={}, token={:?}", id, token);
    }
}
```

### Structured content (function calls, tools, schemas)

```rust
use gemini_tokenizer::{LocalTokenizer, Content, Part, CountTokensConfig,
    Tool, FunctionDeclaration, Schema};
use std::collections::HashMap;

let tokenizer = LocalTokenizer::new("gemini-2.5-pro")
    .expect("failed to load tokenizer");

// Content with text
let contents = vec![Content {
    role: Some("user".to_string()),
    parts: Some(vec![Part {
        text: Some("What is the weather?".to_string()),
        ..Default::default()
    }]),
}];

// Tool definitions via CountTokensConfig
let config = CountTokensConfig {
    tools: Some(vec![Tool {
        function_declarations: Some(vec![FunctionDeclaration {
            name: Some("get_weather".to_string()),
            description: Some("Gets the current weather for a city".to_string()),
            parameters: Some(Schema {
                schema_type: Some("OBJECT".to_string()),
                properties: Some({
                    let mut props = HashMap::new();
                    props.insert("city".to_string(), Schema {
                        schema_type: Some("STRING".to_string()),
                        description: Some("The city name".to_string()),
                        ..Default::default()
                    });
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

let result = tokenizer.count_tokens(contents.as_slice(), Some(&config));
println!("Total tokens: {}", result.total_tokens);
```

### With system instruction

```rust
use gemini_tokenizer::{LocalTokenizer, Content, Part, CountTokensConfig};

let tokenizer = LocalTokenizer::new("gemini-2.5-pro")
    .expect("failed to load tokenizer");

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

let result = tokenizer.count_tokens(contents.as_slice(), Some(&config));
println!("{}", result);
```

## How token counting works

The tokenizer extracts countable text from structured objects following the exact
same rules as the Google Python SDK's `_TextsAccumulator`:

| Content type | What gets counted |
|---|---|
| Text parts | The text string itself |
| Function calls | Function name + all arg keys + all string arg values (recursive) |
| Function responses | Function name + all response keys + all string response values (recursive) |
| Tool declarations | Function name + description + recursive schema traversal |
| Schemas | Format, description, enum values, required fields, property keys, nested schemas |

Numbers, booleans, and null values in function arguments are **not** counted
(matching the Python SDK behavior).

Each extracted text segment is tokenized independently and the counts are summed.

## Supported models

All supported models use the same underlying Gemma 3 tokenizer. The model name is
validated against the same list used by the Python SDK:

- `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- `gemini-2.0-flash`, `gemini-2.0-flash-lite`
- `gemini-2.5-pro-preview-06-05`, `gemini-2.5-pro-preview-05-06`, `gemini-2.5-pro-exp-03-25`
- `gemini-live-2.5-flash`
- `gemini-2.5-flash-preview-05-20`, `gemini-2.5-flash-preview-04-17`
- `gemini-2.5-flash-lite-preview-06-17`
- `gemini-2.0-flash-001`, `gemini-2.0-flash-lite-001`
- `gemini-3-pro-preview`

Use `gemini_tokenizer::supported_models()` to get the full list programmatically.

## Provenance and attribution

This crate is a Rust port of the tokenization logic from the official
**[Google Python GenAI SDK](https://github.com/googleapis/python-genai)** (v1.6.20),
specifically:

- **Text accumulation** — `_TextsAccumulator` class from `google/genai/local_tokenizer.py`
- **Model mapping** — `_GEMINI_MODELS_TO_TOKENIZER_NAMES` and
  `_GEMINI_STABLE_MODELS_TO_TOKENIZER_NAMES` from `google/genai/_local_tokenizer_loader.py`
- **Token-to-bytes conversion** — `_token_str_to_bytes` and `_parse_hex_byte` from
  `google/genai/local_tokenizer.py`

The embedded SentencePiece model file (`gemma3_cleaned_262144_v2.spiece.model`) is from
**[google/gemma_pytorch](https://github.com/google/gemma_pytorch)** at commit
`014acb7ac4563a5f77c76d7ff98f31b568c16508`, with SHA-256 hash:

```
1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c
```

Both upstream projects are licensed under the Apache License, Version 2.0.
See the [NOTICE](NOTICE) file for full attribution details.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

This crate contains code derived from [googleapis/python-genai](https://github.com/googleapis/python-genai)
(Copyright 2025 Google LLC, Apache-2.0) and embeds a tokenizer model from
[google/gemma_pytorch](https://github.com/google/gemma_pytorch)
(Copyright 2024 Google LLC, Apache-2.0).
