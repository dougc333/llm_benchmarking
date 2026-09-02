# Results

- `raw/`: upstream engine benchmark output (ignored by default)
- `normalized/`: canonical, reviewable JSON records suitable for Git
- `training/`: training wrapper logs and summaries (ignored by default)
- `profiles/`: profiler captures (ignored by default)

Normalize a native result with `enginebench results normalize`, then export all
normalized records to the dashboard with `enginebench results export-dashboard`.
Before committing a normalized record, verify that it contains no prompts,
credentials, signed URLs, internal hostnames, or customer data.
