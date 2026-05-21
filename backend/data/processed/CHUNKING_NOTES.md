# Chunking Output Notes

Module 3 converts `LoadedDocument` objects into `DocumentChunk` objects.

## Strategy

- Markdown headings create natural sections.
- Plain text and PDF output become one section if no headings are present.
- Long sections are split with character-based fallback.
- Fallback splitting prefers paragraph, line, and sentence boundaries.
- Chunk overlap is used for long sections so nearby context is not lost.

## Chunk Shape

Each chunk contains:

```text
id
content
metadata
```

Metadata preserves document-level fields:

```text
document_id
chunk_index
source
title
file_name
file_path
file_type
section_title
policy_type
department
country
employee_type
access_level
updated_at
start_char
end_char
```

## Why This Matters

These fields support future modules:

- citations in chat answers
- metadata filtering by country, employee type, access level
- vector-store records
- audit/debug logs
- evaluation of retrieval quality
