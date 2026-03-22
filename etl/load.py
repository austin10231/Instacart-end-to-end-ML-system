from sqlalchemy import create_engine
from .config import DB_URL

def load_table(df, table_name: str):
    engine = create_engine(DB_URL)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
