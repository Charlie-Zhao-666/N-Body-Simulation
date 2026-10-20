# Export the run-history snapshot

`export_run_history.py` updates `data/run-history/` from a local simulation archive.
It never writes to the source archive and never commits or uploads anything.

Requires Python 3.9+ and Pillow (`python -m pip install Pillow`). On Windows,
install `tzdata` too if the Python environment does not provide time-zone data.

From the repository root, run:

```text
python tools/export_run_history.py --source "PATH_TO_LOCAL_ARCHIVE"
```

Replace `PATH_TO_LOCAL_ARCHIVE` with the directory containing `runs.dat`,
`index.html`, PNG figures and metadata. The repository can be on a different
local disk from the simulation source code. An optional `--image-cache`
argument points to previously compressed WebP images; cached pixels are
checked against each original before reuse.

The exporter creates lossless WebP figures, updates the compact README and
complete `runs.md` table, and rewrites Plot paths in the published copies.
`runs.dat` remains aligned plain text; use `runs.md` for clickable links on GitHub.
Existing published PNGs are removed only after their replacement pixels match.
Historical versions remain in Git history after committing the export.
Numerical values are preserved. Comparison-table line endings and trailing
padding are normalized. Raw NPZ trajectories are not copied.

Validation checks table equality, preservation of previously published rows,
image pixels, relative Plot targets and unchanged local source files.
The export is a snapshot: rerun it when publishing new experiments, review
the changes, and then commit and push them using Git.
