"""
PostgreSQL helpers for storing spam prediction results.

Uses the DATABASE_URL environment variable, for example:
postgresql://user:password@host:5432/dbname?sslmode=require
"""

import os
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor


SORT_OPTIONS = {
    "newest": "created_at DESC",
    "oldest": "created_at ASC",
    "highest_spam": "spam_probability DESC, created_at DESC",
}


def normalize_sort(sort):
    """Validate and normalize sort mode."""
    value = (sort or "newest").strip().lower()
    if value not in SORT_OPTIONS:
        raise ValueError(
            "Invalid sort value. Use one of: newest, oldest, highest_spam"
        )
    return value


def get_database_url():
    """Read database URL from environment."""
    return os.getenv("DATABASE_URL", "").strip()


def get_connection():
    """Create a PostgreSQL connection using DATABASE_URL."""
    database_url = get_database_url()
    if not database_url:
        raise ValueError("DATABASE_URL is not set.")

    return psycopg2.connect(database_url)


def test_connection():
    """Test DB connectivity and return server details for diagnostics."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT current_database() AS database, version() AS version")
            info = cursor.fetchone()
            return {
                "database": info["database"],
                "version": info["version"],
            }


def init_db():
    """Create table for prediction logs if it does not already exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id BIGSERIAL PRIMARY KEY,
        message TEXT NOT NULL,
        predicted_label VARCHAR(10) NOT NULL,
        spam_probability DOUBLE PRECISION NOT NULL,
        ham_probability DOUBLE PRECISION NOT NULL,
        risk_level VARCHAR(20) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(create_table_sql)
        conn.commit()


def save_prediction(message, predicted_label, spam_probability, ham_probability, risk_level):
    """Insert one prediction row and return inserted row id."""
    insert_sql = """
    INSERT INTO prediction_logs (
        message,
        predicted_label,
        spam_probability,
        ham_probability,
        risk_level,
        created_at
    )
    VALUES (%s, %s, %s, %s, %s, %s)
    RETURNING id;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                insert_sql,
                (
                    message,
                    predicted_label,
                    float(spam_probability),
                    float(ham_probability),
                    risk_level,
                    datetime.now(timezone.utc),
                ),
            )
            row_id = cursor.fetchone()[0]
        conn.commit()

    return row_id


def get_prediction_logs(page=1, page_size=10, search="", sort="newest"):
    """Fetch recent prediction logs with pagination and optional text search."""
    page = max(1, int(page))
    page_size = max(1, min(100, int(page_size)))
    offset = (page - 1) * page_size
    sort = normalize_sort(sort)
    order_by_clause = SORT_OPTIONS[sort]

    search = (search or "").strip()

    if search:
        query_sql = f"""
        SELECT
            id,
            message,
            predicted_label,
            spam_probability,
            ham_probability,
            risk_level,
            created_at
        FROM prediction_logs
        WHERE message ILIKE %s OR predicted_label ILIKE %s OR risk_level ILIKE %s
        ORDER BY {order_by_clause}
        LIMIT %s OFFSET %s;
        """
        query_params = (f"%{search}%", f"%{search}%", f"%{search}%", page_size, offset)
    else:
        query_sql = f"""
        SELECT
            id,
            message,
            predicted_label,
            spam_probability,
            ham_probability,
            risk_level,
            created_at
        FROM prediction_logs
        ORDER BY {order_by_clause}
        LIMIT %s OFFSET %s;
        """
        query_params = (page_size, offset)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query_sql, query_params)
            rows = cursor.fetchall()

    # Convert datetimes to ISO strings for JSON responses.
    logs = []
    for row in rows:
        row_dict = dict(row)
        if row_dict.get("created_at") is not None:
            row_dict["created_at"] = row_dict["created_at"].isoformat()
        logs.append(row_dict)

    return logs


def get_prediction_logs_for_export(search="", sort="newest", max_rows=10000):
    """Fetch all filtered logs for CSV export, capped by max_rows."""
    sort = normalize_sort(sort)
    max_rows = max(1, min(50000, int(max_rows)))
    order_by_clause = SORT_OPTIONS[sort]
    search = (search or "").strip()

    if search:
        query_sql = f"""
        SELECT
            id,
            message,
            predicted_label,
            spam_probability,
            ham_probability,
            risk_level,
            created_at
        FROM prediction_logs
        WHERE message ILIKE %s OR predicted_label ILIKE %s OR risk_level ILIKE %s
        ORDER BY {order_by_clause}
        LIMIT %s;
        """
        query_params = (f"%{search}%", f"%{search}%", f"%{search}%", max_rows)
    else:
        query_sql = f"""
        SELECT
            id,
            message,
            predicted_label,
            spam_probability,
            ham_probability,
            risk_level,
            created_at
        FROM prediction_logs
        ORDER BY {order_by_clause}
        LIMIT %s;
        """
        query_params = (max_rows,)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query_sql, query_params)
            rows = cursor.fetchall()

    logs = []
    for row in rows:
        row_dict = dict(row)
        if row_dict.get("created_at") is not None:
            row_dict["created_at"] = row_dict["created_at"].isoformat()
        logs.append(row_dict)

    return logs


def count_prediction_logs(search=""):
    """Count logs for pagination, with optional text search filter."""
    search = (search or "").strip()

    if search:
        count_sql = """
        SELECT COUNT(*) AS total
        FROM prediction_logs
        WHERE message ILIKE %s OR predicted_label ILIKE %s OR risk_level ILIKE %s;
        """
        params = (f"%{search}%", f"%{search}%", f"%{search}%")
    else:
        count_sql = "SELECT COUNT(*) AS total FROM prediction_logs;"
        params = None

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            if params is None:
                cursor.execute(count_sql)
            else:
                cursor.execute(count_sql, params)
            row = cursor.fetchone()

    return int(row["total"])
