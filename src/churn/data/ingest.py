"""Load the raw CSV into PostgreSQL.

Requires a running Postgres instance reachable via the settings in .env.
See EXECUTION_GUIDE.docx for how to start one locally with Docker.
"""

from __future__ import annotations

import pandas as pd

from churn.config import PROJECT_ROOT, get_settings
from churn.db.session import get_engine, init_db


def load_csv_to_postgres() -> int:
    settings = get_settings()
    raw_path = PROJECT_ROOT / settings["data"]["raw_path"]
    table_name = settings["database"]["table_name"]

    df = pd.read_csv(raw_path)

    init_db()
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into table '{table_name}'")
    return len(df)


if __name__ == "__main__":
    load_csv_to_postgres()
