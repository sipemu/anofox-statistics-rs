# Agent Instructions

Issues are tracked as GitHub issues (`gh issue list`).

## Validation

**R is always the reference.** Every statistic and p-value is validated
against R (base R or a CRAN package): an `R/generate_*_refs.R` script writes
reference values to `R/data/`, and `tests/` checks them with
`assert_relative_eq!`. Other implementations (scipy, evidently, ...) may be
cross-checked, but where they differ from R, match R and document the
difference.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create GitHub issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Reference or close finished issues
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
