//! Minimal hand-written prost message definitions for SCIP.
//!
//! Covers the subset of `scip.proto` that Axil needs to ingest:
//! `Index`, `Document`, `Occurrence`, `SymbolInformation`, `Relationship`.
//! Full spec: <https://github.com/sourcegraph/scip/blob/main/scip.proto>.
//!
//! We deliberately avoid `prost-build` + a build script so the crate stays
//! a pure-Rust library with no protoc/proto-file build-time dependency.

use prost::Message;

#[derive(Clone, PartialEq, Message)]
pub struct Index {
    #[prost(message, optional, tag = "1")]
    pub metadata: Option<Metadata>,
    #[prost(message, repeated, tag = "2")]
    pub documents: Vec<Document>,
    #[prost(message, repeated, tag = "3")]
    pub external_symbols: Vec<SymbolInformation>,
}

#[derive(Clone, PartialEq, Message)]
pub struct Metadata {
    #[prost(int32, tag = "1")]
    pub version: i32,
    #[prost(message, optional, tag = "2")]
    pub tool_info: Option<ToolInfo>,
    #[prost(string, tag = "3")]
    pub project_root: String,
    #[prost(int32, tag = "4")]
    pub text_document_encoding: i32,
}

#[derive(Clone, PartialEq, Message)]
pub struct ToolInfo {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(string, tag = "2")]
    pub version: String,
    #[prost(string, repeated, tag = "3")]
    pub arguments: Vec<String>,
}

#[derive(Clone, PartialEq, Message)]
pub struct Document {
    #[prost(string, tag = "4")]
    pub language: String,
    #[prost(string, tag = "1")]
    pub relative_path: String,
    #[prost(message, repeated, tag = "2")]
    pub occurrences: Vec<Occurrence>,
    #[prost(message, repeated, tag = "3")]
    pub symbols: Vec<SymbolInformation>,
    #[prost(string, tag = "5")]
    pub text: String,
    /// `position_encoding`, `0` = UTF-8 (default).
    #[prost(int32, tag = "6")]
    pub position_encoding: i32,
}

#[derive(Clone, PartialEq, Message)]
pub struct Occurrence {
    /// `[start_line, start_character, end_line, end_character]` or
    /// `[start_line, start_character, end_character]` (same-line form).
    #[prost(int32, repeated, packed = "true", tag = "1")]
    pub range: Vec<i32>,
    #[prost(string, tag = "2")]
    pub symbol: String,
    /// Bitfield. See `SymbolRole`.
    #[prost(int32, tag = "3")]
    pub symbol_roles: i32,
    #[prost(string, repeated, tag = "4")]
    pub override_documentation: Vec<String>,
    #[prost(int32, tag = "5")]
    pub syntax_kind: i32,
    #[prost(int32, repeated, packed = "true", tag = "7")]
    pub enclosing_range: Vec<i32>,
}

/// SCIP symbol-role bitfield values. Subset we care about.
pub mod symbol_role {
    pub const DEFINITION: i32 = 0x1;
    pub const IMPORT: i32 = 0x2;
    pub const WRITE_ACCESS: i32 = 0x4;
    pub const READ_ACCESS: i32 = 0x8;
    pub const GENERATED: i32 = 0x10;
    pub const TEST: i32 = 0x20;
    pub const FORWARD_DEFINITION: i32 = 0x40;
}

#[derive(Clone, PartialEq, Message)]
pub struct SymbolInformation {
    #[prost(string, tag = "1")]
    pub symbol: String,
    #[prost(string, repeated, tag = "3")]
    pub documentation: Vec<String>,
    #[prost(message, repeated, tag = "4")]
    pub relationships: Vec<Relationship>,
    #[prost(int32, tag = "5")]
    pub kind: i32,
    #[prost(string, tag = "6")]
    pub display_name: String,
    #[prost(string, tag = "8")]
    pub enclosing_symbol: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct Relationship {
    #[prost(string, tag = "1")]
    pub symbol: String,
    #[prost(bool, tag = "2")]
    pub is_reference: bool,
    #[prost(bool, tag = "3")]
    pub is_implementation: bool,
    #[prost(bool, tag = "4")]
    pub is_type_definition: bool,
    #[prost(bool, tag = "5")]
    pub is_definition: bool,
}

/// Parse a SCIP index from raw protobuf bytes.
pub fn decode_index(bytes: &[u8]) -> Result<Index, prost::DecodeError> {
    Index::decode(bytes)
}
