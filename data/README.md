# Data Directory

Private collected datasets are intentionally not committed to GitHub.

Use this folder structure locally:

```text
data/
`-- private/
    `-- landmarks_all.csv
```

The default Python data collector writes to `data/private/landmarks_all.csv`, and `.gitignore` excludes that path. If exported tester sessions are collected, keep them in `exports/` or `user-testing-exports/`, which are also ignored.
