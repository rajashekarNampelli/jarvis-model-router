# Git Commit Instructions

## When to Commit
When the user says "commit my changes", "commit and push", or similar.

## Git Identity

Before any commit, verify git identity is properly configured:

1. **Check** — `git config user.name` and `git config user.email`
2. **If missing or auto-guessed** (e.g. `user@hostname.local`), set it:
   - `git config --global user.name "Rajashekar Nampelli"`
   - `git config --global user.email "rajashekar.nampelli@gmail.com"`
3. **If the last commit has a bad identity** — `git commit --amend --reset-author --no-edit`
4. **Then force-push** — `git push --force-with-lease`

Never commit with an auto-guessed email like `raja@Rajashekars-MacBook-Pro.local`.

## Pre-flight Checks

1. **Run unit tests** — `pytest -v` — all must pass before committing
2. **Run linter** — `ruff check --select F401,F841,F811 jarvis_model_router/ smoke_test.py` — zero errors
3. **Check git status** — `git status --short` — see what's changed

## Staging

1. **Stage all changed files** — `git add -A`
2. **Review the diff** — `git diff --cached --stat` — know what's going in
3. **Check for untracked files** that should be committed (new files, config, etc.)
4. **Check for files that should NOT be committed** — secrets, .env, __pycache__, .venv, IDE files, test result artifacts. These should be in `.gitignore`

## Commit Message Format

Use a structured commit message with a **short subject line** and **categorized body**:

```
<scope>: <short summary in imperative mood>

Enhancements:
- <what was added or improved>

Dead code / cleanup:
- <what was removed and why>

Bug fixes:
- <what was broken and how it was fixed>

Other:
- <anything else worth noting>
- All N unit tests pass
```

### Rules for the commit message
- **Subject line**: scope + colon + short summary, lowercase, imperative mood (e.g. "add feature" not "added feature")
- **Scope**: use `v1`, `v2`, `classifier`, `api`, `infra`, `cleanup`, etc. — whatever fits
- **Body categories**: use `Enhancements:`, `Dead code / cleanup:`, `Bug fixes:`, `Other:` — include only categories that have items
- **Each bullet**: start with a verb (Add, Remove, Replace, Fix, Simplify, Update), be specific about what file/feature changed
- **Last line of Other**: always include test pass count (e.g. "All 27 unit tests pass")
- **No emojis** in commit messages

## Push

After committing:
1. `git push` — push to remote
2. Report the commit hash and branch to the user

## Example Commit Messages

Good:
```
v2: LLM classifier, connection pooling, SSE streaming, dead code removal

Enhancements:
- Replace keyword-based classifier with LLM-based classifier (llama3)
- Shared httpx.AsyncClient with connection pooling
- SSE streaming format (text/event-stream with data: prefix)

Dead code / cleanup:
- Remove CODE_KEYWORDS, REASONING_KEYWORDS (40+ keyword entries)
- Remove _keyword_classify() fallback function
- Remove unused python-dotenv, duplicate httpx from requirements

Other:
- Add smoke_test.py with 8 prompts, routing accuracy tracking
- All 27 unit tests pass
```

Bad (do NOT do this):
```
fixed stuff
```
```
updates
```
```
 Major improvements to the codebase!!! 
```

## Flow Summary

```
check git identity -> pytest -> ruff -> git add -A -> git diff --cached --stat -> commit -> push -> report
```
