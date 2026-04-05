# .claude/CLAUDE.md - Operational Directives

## Workflow

1. **Plan first** - Write plans to `.claude_plans/` with checkpoints, tasks, and success criteria. Get approval before implementing.
2. **Work incrementally** - One small task at a time. Test after each significant change.
3. **Complete on finish** - Update the plan, verify end-to-end, mark tasks done.

## Rules

- Complete working implementations only - no stubs, TODOs, or placeholders
- No orphan files in root - sort everything into the appropriate directory
- Tests go in `tests/`
- Reference `.claude_prompts/course_creator.md` for the full course creation framework

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
