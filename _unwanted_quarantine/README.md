# `_unwanted_quarantine`

This folder holds material **relocated out of canonical project paths** for team review before permanent removal from the repo.

- **Purpose:** Preserve a reversible audit trail (via `git log`/`git mv` history where applicable) instead of silently deleting duplicated or sensitive-path noise.
- **Do not:** Add new secrets here, or paste `.env` contents into tracked files.

## Current contents

| Path | Reason | Date archived |
|------|--------|---------------|
| `app-frontend-admin-Copy-archive/` | Duplicate Next.js tree (`app/frontend-admin - Copy`); canonical app is `app/frontend-admin`. | 2026-04-30 |

After sign-off: delete `_unwanted_quarantine/` entirely in a dedicated commit (`git rm -r _unwanted_quarantine`), or keep this archive briefly for reference—team policy decides.

**This branch keeps the archive in place** until that review completes (no Phase-E delete bundled with the hygiene commit).
