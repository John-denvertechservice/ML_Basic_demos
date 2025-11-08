# Tiny ML Playground – Agent Contract

## Project Snapshot
- **Mission**: Ship playful ML experiments that run fast on a 2019 MacBook Pro i5 with ~11 GB free space.
- **Scope**: Python-only, CPU-friendly demos (sklearn, tiny PyTorch); no massive datasets, no GPU assumptions.
- **Out of Scope**: Cloud/GPU jobs, installs >1 GB, long-running training, closed-source dependencies.

## Environment Facts
- Hardware: 2019 MBP, Intel i5 CPU, limited storage (watch disk usage; target <1–2 GB for this project).
- OS: macOS (user will provide version if a task needs it).
- Tooling already installed: Homebrew, system Python interpreter (exact version TBD per task).
- Network: restricted by default; request approval before fetching large files or packages.

## Shared Workflow (Codex · Claude · Cursor)
1. **Plan First** – Outline 2–5 short steps (<=7 words each); only one `in_progress` at a time.
2. **Stay Focused** – Touch only files called out in the plan; keep diffs minimal and on-topic.
3. **Explain Moves** – Reference files with paths + start line (`path/to/file.py:42`) when summarizing.
4. **Validate** – Run the smallest useful check (tests, lint, script). Record the exact commands + outcome.
5. **Handoff** – Summarize what changed, why it matters, and list next steps or open questions.

## Coding Conventions
- Python 3.10+ preferred when available.
- Use `requirements.txt` with pinned versions; prefer CPU wheels (e.g., `torch==<cpu-build>`).
- Default layout suggestion:
  ```
  playground/
    cli.py
    data/
    models/
    viz/
  notebooks/
  plans/
  ```
- Keep datasets tiny (<=100 MB total). Leverage sklearn/tiny torchvision datasets or synthetic generators.
- Add concise comments only for non-obvious logic.

## Tooling & Safety
- **PyTorch** is the default DL stack; TensorFlow optional later.
- Avoid network-heavy installs unless cleared; document any new dependency (reason + size impact).
- No destructive git commands (`reset --hard`, etc.) unless explicitly requested.
- Secrets: none should exist; if encountered, stop and ask.

## Testing & Performance
- Prefer fast unit/functional tests (seconds). For training loops, cap default epochs (<=3) and dataset subsets.
- Document any skipped tests + justification.
- Ensure commands complete comfortably on CPU-only environment.

## Decision Log Template
Record significant choices in this section (most recent first).

| Date | Decision | Options Considered | Rationale |
|------|----------|--------------------|-----------|
| TBD  | –        | –                  | –         |

## Open Questions / TODOs
- [ ] Confirm installed Python version and create `requirements.txt`.
- [ ] Decide on initial PyTorch example (e.g., MNIST subset vs. synthetic spirals).
- [ ] Define minimal CLI entry point (`python -m playground ...`?).
