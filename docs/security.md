# KIBAD Security

## Principles

KIBAD is designed for use in environments with strict data security requirements.
All data processing happens locally. There are no external network calls, no
telemetry, and no data exfiltration paths.

## Network Isolation

### No outbound connections

- Streamlit telemetry is disabled: `gatherUsageStats = false` in
  `.streamlit/config.toml`.
- No analytics, tracking, or crash-reporting services.
- No CDN-hosted assets -- all static resources are bundled locally.
- No auto-update mechanisms.

### Verification

To verify KIBAD makes no network calls, monitor outbound traffic while running:

```bash
# macOS
sudo lsof -i -P | grep python

# Linux
ss -tnp | grep python
```

The only listening socket should be the Streamlit server on localhost.

## Data Locality

All data remains on the local machine:

- Uploaded files are read into memory as pandas DataFrames and stored in
  `st.session_state`. They are never written to disk unless the user explicitly
  exports them.
- Temporary files (if any) are created in the system temp directory and deleted
  after use.
- Generated reports (PDF/HTML) are offered as downloads, not stored permanently.
- PostgreSQL connections are made directly to the user-specified host; no proxy
  or intermediary is involved.

## Audit Logging

The `core/audit.py` module maintains an append-only in-session audit log that
records every significant operation:

| Event Type        | Logged Fields                                       |
|-------------------|-----------------------------------------------------|
| File load         | Filename, format, row count, column count, timestamp |
| Data transform    | Operation name, parameters, rows affected, timestamp |
| Analysis run      | Test/model name, parameters, result summary, timestamp|
| Export            | Format (CSV/PDF/HTML), filename, timestamp           |
| Error             | Operation, error message, traceback summary, timestamp|

The audit log is available on the Report page and can be exported as part of
the analysis report. It provides a full trace of how results were derived from
raw data.

## File Handling

### Accepted formats

| Format   | Extension       | Library     |
|----------|-----------------|-------------|
| CSV      | .csv            | pandas      |
| Excel    | .xlsx           | openpyxl    |
| Excel    | .xls            | xlrd        |
| Parquet  | .parquet        | pyarrow     |

### Validation checks

- **Extension whitelist**: only the formats listed above are accepted. Files with
  other extensions are rejected with a clear error message.
- **Size limit**: files larger than 200 MB are rejected to prevent memory
  exhaustion. The limit is configurable but defaults to 200 MB.
- **Content validation**: after loading, the file must parse into a valid
  DataFrame with at least one column and one row.
- **Encoding detection**: CSV files are read with UTF-8 by default, with
  fallback to cp1251 (common for Russian-locale files) and latin-1.

### What is NOT done

- No file is executed as code.
- No macros in Excel files are evaluated.
- No embedded objects or external references in Excel are followed.

## Input Sanitization

### Numeric parsing

All numeric parsing uses safe coercion:

```python
pd.to_numeric(series, errors="coerce")  # invalid values become NaN
```

- No `eval()` or `exec()` is used on user-provided strings.
- Currency symbols, whitespace, and thousand separators are stripped with
  explicit regex patterns before parsing.
- Results are always checked for NaN after conversion.

### Date parsing

- Dates are parsed with explicit format strings or `pd.to_datetime` with
  `errors="coerce"`.
- Bounds checking: dates before 1900-01-01 or after 2100-12-31 trigger a
  warning.
- No timezone assumptions -- all dates are treated as naive unless the user
  specifies a timezone.

### KPI Formulas

Users can define custom KPI formulas on the Explore page. These are evaluated
in a restricted namespace:

```python
ALLOWED_NAMES = {
    "abs", "round", "min", "max", "sum", "mean", "std",
    "log", "exp", "sqrt", "pow",
}
```

- No access to `__builtins__`, `import`, `open`, `exec`, `eval`, or any
  system functions.
- The formula string is parsed with Python's `ast` module to verify it contains
  only allowed operations before evaluation.
- Maximum formula length: 500 characters.

## PostgreSQL Connection Security

- Database credentials are entered by the user in the UI and stored only in
  `st.session_state` for the duration of the session. They are never written to
  disk, logged, or included in reports.
- All queries use SQLAlchemy's parameterized query interface to prevent SQL
  injection:

```python
with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM :table"), {"table": table_name})
```

- Connection strings are built programmatically -- never concatenated from
  user input.
- The application uses read-only queries (`SELECT` only). No `INSERT`,
  `UPDATE`, `DELETE`, or DDL statements are issued.

## Session Security

- Each Streamlit session is isolated. There is no shared state between users.
- Session data is stored in memory and garbage-collected when the session ends.
- No cookies, tokens, or authentication mechanisms are used (Streamlit handles
  session management internally).

## Threat Model Summary

| Threat                    | Mitigation                                       |
|---------------------------|--------------------------------------------------|
| Data exfiltration         | No network calls, all data local                 |
| Malicious file upload     | Extension whitelist, size limit, no macro eval    |
| SQL injection             | Parameterized queries via SQLAlchemy              |
| Code injection via KPI    | AST-based formula validation, restricted namespace|
| Memory exhaustion         | 200 MB file size limit                           |
| Credential leakage        | In-memory only, never logged or exported         |
| Cross-session data leak   | Isolated session state, no shared storage        |
