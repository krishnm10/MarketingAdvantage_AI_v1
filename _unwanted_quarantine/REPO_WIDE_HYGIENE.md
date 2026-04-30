# Repo-wide accidental tracking

Historical commits had:

- **`python .venv`** (~25k files) and **root `node_modules/`** (~3k files) indexed by Git — they should never ship in application source control.

**What we changed:** `git rm -r --cached .venv` and `git rm -r --cached node_modules`.

**Working tree:** Unchanged — your interpreter and npm tree stay on disk if they already exist locally.

**Fresh clone:** Developers run `python -m venv .venv` (or restore their env workflow) and `npm install` at the repo root if that tooling is needed for root-level JS tools.
