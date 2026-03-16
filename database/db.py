import pandas as pd
from sqlalchemy import create_engine, text
from config.settings import DB_URL
from utils.logger import logger

engine = create_engine(DB_URL, pool_size=5, max_overflow=10, pool_pre_ping=True)


def get_engine():
    return engine


def get_connection():
    return engine.connect()


def execute_sql(sql: str, params: dict = None):
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        conn.commit()
        return result


def read_sql(query: str, params: dict = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def write_df(df: pd.DataFrame, table: str, if_exists: str = "append"):
    df.to_sql(table, engine, if_exists=if_exists, index=False, method="multi")


def init_db():
    """Run the schema.sql to initialize all tables and hypertables."""
    import os
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        sql = f.read()
    with engine.connect() as conn:
        for statement in sql.split(";"):
            stmt = statement.strip()
            if stmt:
                try:
                    conn.execute(text(stmt))
                except Exception as e:
                    logger.warning(f"Schema statement skipped: {e}")
        conn.commit()
    logger.info("Database schema initialized.")