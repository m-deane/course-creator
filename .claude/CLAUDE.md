# .claude/CLAUDE.md - Operational Directives

## Workflow Rules

- Never use mock data, results, or workarounds
- Write plans and progress tracking to `.claude_plans/`
- Write all tests to `tests/`
- No orphan files in root - sort everything into the appropriate directory
- Complete working implementations only - no stubs, TODOs, or placeholders

## Communication Style

- Be direct and concise
- No hedging language ("might", "could potentially", "perhaps")
- No social validation ("Great question!", "You're absolutely right!")
- Skip restating requirements already provided
- When multiple approaches exist, recommend one clearly

## Code Style (Python)

- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Files: `snake_case.py`
