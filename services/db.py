"""
services/db.py – PostgreSQL connector using SQLAlchemy + psycopg2.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def build_connection_string(
    host: str,
    port: int | str,
    database: str,
    user: str,
    password: str,
) -> str:
    """Build a SQLAlchemy PostgreSQL connection string.

    Parameters
    ----------
    host:
        Database hostname or IP.
    port:
        Port number (default 5432).
    database:
        Database name.
    user:
        Database user.
    password:
        Database password.

    Returns
    -------
    str
        SQLAlchemy connection URL.
    """
    from urllib.parse import quote_plus
    pw_encoded = quote_plus(str(password))
    return f"postgresql+psycopg2://{user}:{pw_encoded}@{host}:{port}/{database}"


def test_connection(
    host: str,
    port: int | str,
    database: str,
    user: str,
    password: str,
) -> tuple[bool, str]:
    """Test a PostgreSQL connection.

    Parameters
    ----------
    host, port, database, user, password:
        Connection parameters.

    Returns
    -------
    tuple[bool, str]
        (success, message)
    """
    try:
        from sqlalchemy import create_engine, text
        conn_str = build_connection_string(host, port, database, user, password)
        engine = create_engine(conn_str, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connection successful."
    except ImportError:
        return False, "sqlalchemy or psycopg2 not installed."
    except Exception as exc:
        return False, f"Connection failed: {exc}"


def query_to_dataframe(
    host: str,
    port: int | str,
    database: str,
    user: str,
    password: str,
    query: str,
    chunksize: int | None = None,
) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame.

    Parameters
    ----------
    host, port, database, user, password:
        Connection parameters.
    query:
        SQL SELECT query string.
    chunksize:
        If set, read in chunks and concatenate (for large result sets).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ImportError
        If sqlalchemy or psycopg2 is not installed.
    RuntimeError
        On connection or query errors.
    """
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise ImportError("sqlalchemy and psycopg2 are required.") from exc

    conn_str = build_connection_string(host, port, database, user, password)
    try:
        engine = create_engine(conn_str)
        if chunksize:
            chunks = pd.read_sql(query, engine, chunksize=chunksize)
            return pd.concat(list(chunks), ignore_index=True)
        return pd.read_sql(query, engine)
    except Exception as exc:
        raise RuntimeError(f"Query execution failed: {exc}") from exc


def list_tables(
    host: str,
    port: int | str,
    database: str,
    user: str,
    password: str,
    schema: str = "public",
) -> list[str]:
    """List all tables in the specified schema.

    Parameters
    ----------
    host, port, database, user, password:
        Connection parameters.
    schema:
        PostgreSQL schema name.

    Returns
    -------
    list[str]
        List of table names.
    """
    query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{schema}'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """
    try:
        df = query_to_dataframe(host, port, database, user, password, query)
        return df["table_name"].tolist()
    except Exception:
        return []


def list_schemas(
    host: str,
    port: int | str,
    database: str,
    user: str,
    password: str,
) -> list[str]:
    """List all accessible schemas in the database.

    Returns
    -------
    list[str]
        Schema names, with 'public' first if present.
    """
    query = """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
          AND schema_name NOT LIKE 'pg_%'
        ORDER BY schema_name;
    """
    try:
        df = query_to_dataframe(host, port, database, user, password, query)
        schemas = df["schema_name"].tolist()
        # Put 'public' first
        if "public" in schemas:
            schemas = ["public"] + [s for s in schemas if s != "public"]
        return schemas
    except Exception:
        return ["public"]


def get_table_columns(
    host: str,
    port: int | str,
    database: str,
    user: str,
    password: str,
    table: str,
    schema: str = "public",
) -> list[dict]:
    """Get column metadata for a table.

    Returns
    -------
    list[dict]
        Each dict has keys: name, data_type, is_nullable.
    """
    query = f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = '{table}'
        ORDER BY ordinal_position;
    """
    try:
        df = query_to_dataframe(host, port, database, user, password, query)
        return df.to_dict("records")
    except Exception:
        return []


def get_row_count(
    host: str,
    port: int | str,
    database: str,
    user: str,
    password: str,
    table: str,
    schema: str = "public",
) -> int | None:
    """Get approximate row count for a table via pg_stat_user_tables."""
    query = f"""
        SELECT reltuples::bigint AS row_count
        FROM pg_class
        JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
        WHERE pg_namespace.nspname = '{schema}'
          AND relname = '{table}';
    """
    try:
        df = query_to_dataframe(host, port, database, user, password, query)
        if not df.empty:
            return int(df.iloc[0]["row_count"])
        return None
    except Exception:
        return None


def get_distinct_values(
    host: str,
    port: int | str,
    database: str,
    user: str,
    password: str,
    table: str,
    column: str,
    schema: str = "public",
    limit: int = 100,
) -> list[Any]:
    """Get distinct values for a column (for filter dropdowns)."""
    query = f"""
        SELECT DISTINCT "{column}"
        FROM "{schema}"."{table}"
        WHERE "{column}" IS NOT NULL
        ORDER BY "{column}"
        LIMIT {limit};
    """
    try:
        df = query_to_dataframe(host, port, database, user, password, query)
        return df[column].tolist()
    except Exception:
        return []


def build_nocode_query(
    schema: str,
    table: str,
    columns: list[str] | None = None,
    computed_cols: list[dict] | None = None,   # NEW: [{expr, alias}, ...]
    distinct: bool = False,                     # NEW
    joins: list[dict] | None = None,            # NEW: [{type, schema, table, on_left, on_right}, ...]
    filters: list[dict] | None = None,
    group_by: list[str] | None = None,
    aggregations: list[dict] | None = None,     # NEW: [{func, col, alias}, ...]
    having: str | None = None,                  # NEW
    order_by: list[dict] | None = None,
    limit: int | None = 10000,
) -> str:
    """Build a SQL SELECT query from NoCode builder parameters.

    Parameters
    ----------
    schema:
        Schema name.
    table:
        Table name.
    columns:
        List of column names to select; None → SELECT *.
    computed_cols:
        List of computed column dicts: {expr, alias}.
    distinct:
        If True, adds DISTINCT keyword after SELECT.
    joins:
        List of JOIN dicts: {type, schema, table, on_left, on_right}.
    filters:
        List of filter dicts: {col, op, val, connector}.
        connector is 'AND' or 'OR' (applies before this condition).
    group_by:
        List of column names to GROUP BY.
    aggregations:
        List of aggregation dicts: {func, col, alias}.
    having:
        HAVING clause expression (without the HAVING keyword).
    order_by:
        List of order dicts: {col, direction} where direction is 'ASC'/'DESC'.
    limit:
        Row limit; None → no LIMIT.

    Returns
    -------
    str
        SQL query string.
    """
    # SELECT clause
    select_keyword = "SELECT DISTINCT" if distinct else "SELECT"

    if columns:
        select_parts = [f'"{c}"' for c in columns]
    else:
        select_parts = ["*"]

    # Computed columns
    if computed_cols:
        for cc in computed_cols:
            expr = cc.get("expr", "").strip()
            alias = cc.get("alias", "").strip()
            if expr:
                if alias:
                    select_parts.append(f'{expr} AS "{alias}"')
                else:
                    select_parts.append(expr)

    # Aggregation columns
    if aggregations:
        for agg in aggregations:
            func = agg.get("func", "SUM")
            col = agg.get("col", "")
            alias = agg.get("alias", "").strip()
            if col:
                if func == "COUNT DISTINCT":
                    agg_expr = f'COUNT(DISTINCT "{col}")'
                else:
                    agg_expr = f'{func}("{col}")'
                if alias:
                    select_parts.append(f'{agg_expr} AS "{alias}"')
                else:
                    select_parts.append(agg_expr)

    cols_str = ", ".join(select_parts)
    sql = f'{select_keyword} {cols_str}\nFROM "{schema}"."{table}"'

    # JOIN clauses
    if joins:
        for j in joins:
            jschema = j.get("schema", schema)
            jtable = j.get("table", "")
            on_l = j.get("on_left", "")
            on_r = j.get("on_right", "")
            jtype = j.get("type", "INNER JOIN")
            if jtable and on_l and on_r:
                sql += f'\n{jtype} "{jschema}"."{jtable}" ON "{table}"."{on_l}" = "{jtable}"."{on_r}"'

    # WHERE clause
    if filters:
        conditions = []
        for i, f in enumerate(filters):
            col = f.get("col", "")
            op = f.get("op", "=")
            val = f.get("val", "")
            connector = f.get("connector", "AND") if i > 0 else ""

            # Quote string values
            if op in ("IS NULL", "IS NOT NULL"):
                condition = f'"{col}" {op}'
            elif op in ("IN", "NOT IN"):
                # val is comma-separated
                items = [v.strip() for v in str(val).split(",")]
                quoted = ", ".join(f"'{v}'" for v in items)
                condition = f'"{col}" {op} ({quoted})'
            elif isinstance(val, str) and not val.lstrip("-").replace(".", "").isdigit():
                condition = f'"{col}" {op} \'{val}\''
            else:
                condition = f'"{col}" {op} {val}'

            if i == 0:
                conditions.append(condition)
            else:
                conditions.append(f"{connector} {condition}")

        sql += "\nWHERE " + "\n  ".join(conditions)

    # GROUP BY clause
    if group_by:
        gb_str = ", ".join(f'"{c}"' for c in group_by)
        sql += f"\nGROUP BY {gb_str}"

    # HAVING clause
    if having and having.strip():
        sql += f"\nHAVING {having.strip()}"

    # ORDER BY clause
    if order_by:
        ob_parts = [f'"{o["col"]}" {o.get("direction", "ASC")}' for o in order_by]
        sql += "\nORDER BY " + ", ".join(ob_parts)

    # LIMIT clause
    if limit is not None:
        sql += f"\nLIMIT {limit}"

    return sql
