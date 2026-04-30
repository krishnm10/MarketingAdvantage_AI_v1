# `_unwanted_quarantine`

This folder holds material **relocated out of canonical project paths** for team review before permanent removal from the repo.

- **Purpose:** Preserve a reversible audit trail (via `git log`/`git mv` history where applicable) instead of silently deleting duplicated or sensitive-path noise.
- **Do not:** Add new secrets here, or paste `.env` contents into tracked files.

## Current contents

| Path | Reason | Date archived |
|------|--------|---------------|
| `app-frontend-admin-Copy-archive/` | Duplicate Next.js tree (`app/frontend-admin - Copy`); canonical app is `app/frontend-admin`. | 2026-04-30 |
| `duplicate-and-backup-code/app/...` | Windows ` - Copy*.py`, dated `ingestion_service_v2_*` snapshots, phantom/Chroma forks — none are valid import targets (canonical modules kept in place). | 2026-04-30 |

## Index-only hygiene (nothing moved on disk except Git index)

- **`.venv/`** and **`node_modules/`** (repo root): were mistakenly tracked (`git ls-files`). Removed from index with **`git rm -r --cached`** only — your **local** folders remain. Reinstall/use venv & `npm install` as usual on fresh clones. See [REPO_WIDE_HYGIENE.md](./REPO_WIDE_HYGIENE.md).

## Explicitly NOT quarantined (still “real” product surface)

- **`app/core/configs/defau.json`** — valid client config id `defau` (resolved via `client_id.defau`). Not a typo of `default.json`.
- **Other “dead code” candidates** — not bulk-moved without `vulture`/import-graph review; trimming those needs a targeted follow-up PR.

After sign-off: delete `_unwanted_quarantine/` entirely in a dedicated commit (`git rm -r _unwanted_quarantine`), or keep this archive briefly for reference—team policy decides.

**This branch keeps the archive in place** until that review completes (no Phase-E delete bundled with the hygiene commit).
