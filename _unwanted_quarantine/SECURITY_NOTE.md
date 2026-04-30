# Security note: removing `.env` from version control

Stopping tracking of `.env` (and the stray ` - Copy.env`) **will show the previous file contents as deletions in `git diff`** for the commit that lands the change. Assume any shared review channel or forge UI may expose those lines.

**Before opening a public PR or merging:** rotate all keys and secrets that ever appeared in tracked `.env`. Git history retains old blobs until you rewrite history (e.g. `git filter-repo`); this cleanup only removes them from the latest tree.
