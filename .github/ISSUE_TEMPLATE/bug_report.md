---
name: Bug report
about: Report a bug or workflow integration issue
title: "[Bug]: "
labels: bug
---

## Summary

Describe what happened and what you expected.

## Environment

- `mvdirector` version (`mvd --version` or commit hash):
- Python version:
- OS / Docker / ComfyUI hosting (local, home LAN, or cloud):
- ComfyUI version, workflow & models used:
- LLM provider & model:

## Diagnostic Output

Please paste the output of both commands below (redacting any tokens or personal paths):

```bash
mvd validate -p <your-project-dir>
mvd llm-check -p <your-project-dir>
```

## Redacted `project.yaml`

```yaml
# Paste project.yaml with any secrets removed
```
