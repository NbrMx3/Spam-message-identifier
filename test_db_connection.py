"""
Quick PostgreSQL connection test.

Usage:
1) Set environment variable DATABASE_URL
2) Run: python test_db_connection.py
"""

from db import test_connection, init_db


def main():
    try:
        info = test_connection()
        print("✓ Database connection successful")
        print(f"  Database: {info['database']}")

        init_db()
        print("✓ Table check successful (prediction_logs is ready)")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")


if __name__ == "__main__":
    main()
