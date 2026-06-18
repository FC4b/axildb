---
name: axil-report
description: "Generate a structured field report about Axil problems encountered during real work"
trigger: when user mentions axil report, axil problems, axil issues, report axil bug, axil slow
---

# Axil Field Report Generator

Generate a structured JSON report about Axil problems encountered during work. This report can be consumed by the Axil source repo's `/axil-diagnose` skill to fix bugs and optimize performance.

## When to Use

- Axil commands are slow (>100ms for simple operations)
- Axil commands return errors or crash
- Search/recall returns unexpected or missing results
- Data corruption or inconsistency suspected
- After any Axil-related issue during normal work

## How to Generate a Report

### Step 1: Run the CLI report generator

```bash
axil report generate --db ./memory.axil
```

This creates a baseline report in `.axil-reports/report-<date>.json`.

### Step 2: Enrich the report with observed problems

After generating, read the report and add problem entries for issues you've observed. Each problem should have:

```json
{
  "type": "error|performance|data|crash",
  "severity": "error|warning|info",
  "component": "storage|vector_search|graph|fts|timeseries|cli",
  "description": "Human-readable description of the problem",
  "command": "exact axil command that triggered it",
  "timing_ms": 230,
  "context": "what was happening when the problem occurred",
  "stacktrace": "if available",
  "reproduction": "steps to reproduce"
}
```

### Step 3: Update the report file

Read the generated report, add your problem observations to the `problems` array, and write it back:

```bash
# Read current report
cat .axil-reports/report-*.json | jq '.'

# The report file can be edited directly — it's just JSON
```

## Report Schema (v1.0)

```json
{
  "version": "1.0",
  "generated_at": "2026-04-01T10:00:00Z",
  "axil_version": "0.1.0",
  "environment": {
    "os": "darwin",
    "arch": "aarch64",
    "features": ["vector", "graph", "fts"]
  },
  "database": {
    "path": "./memory.axil",
    "size_bytes": 12400000,
    "record_count": 1234,
    "tables": {"sessions": 50, "decisions": 200}
  },
  "problems": [
    {
      "type": "performance",
      "severity": "warning",
      "component": "vector_search",
      "description": "axil recall takes >200ms consistently",
      "command": "axil recall 'auth timeout' --top-k 5",
      "timing_ms": 230,
      "context": "10k vectors indexed, mostly in 'decisions' table"
    }
  ],
  "usage_patterns": {
    "most_used_commands": ["recall", "store", "session log"],
    "tables_by_write_frequency": ["logs", "decisions", "sessions"]
  },
  "config": { }
}
```

## Problem Types

| Type | When to Use |
|------|-------------|
| `error` | Command returned an error or non-zero exit code |
| `performance` | Command was unexpectedly slow |
| `data` | Wrong results, missing data, inconsistency |
| `crash` | Panic, segfault, or process killed |

## Severity Levels

| Severity | When to Use |
|----------|-------------|
| `error` | Blocks work, data loss risk |
| `warning` | Degraded experience, workaround exists |
| `info` | Minor issue, nice to fix |

## After Generating

The report is saved to `.axil-reports/` (configurable via `axil.toml` → `dev.reports_dir`).

To send it to the Axil source repo:
1. If using path dependency: `axil report import --from .` (run from the Axil source repo)
2. Manual copy: copy the JSON file to `<axil-repo>/reports/incoming/`
3. The Axil developer then runs `/axil-diagnose` to read and fix reported issues

## Managing Reports

```bash
# List all generated reports
axil report list

# Reports are gitignored by default — they contain project-specific data
```
