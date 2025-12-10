# convert_csvs.py
import pandas as pd
import sqlite3
import argparse
from pathlib import Path
from sqlalchemy import create_engine

def csv_to_sqlite(csv_path: Path, db_path: Path, table_name: str, dtype_map: dict = None):
    print(f"Converting {csv_path} -> {db_path} (table: {table_name})")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(' ', '_').replace('-', '_') for c in df.columns]
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Wrote {len(df)} rows, columns: {list(df.columns)}")

def main(data_dir='data'):
    data_dir = Path(data_dir)

    mappings = [
        (data_dir/'heart.csv', Path('heart_disease.db'), 'heart_disease'),
        (data_dir/'cancer.csv', Path('cancer.db'), 'cancer'),
        (data_dir/'diabetes.csv', Path('diabetes.db'), 'diabetes'),
    ]
    for csv_path, db_path, table_name in mappings:
        if not csv_path.exists():
            print(f"Missing {csv_path}. Please download and put it in {data_dir}.")
            continue
        csv_to_sqlite(csv_path, db_path, table_name)

if __name__ == '__main__':
    import dotenv, os
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='folder with CSV files')
    args = parser.parse_args()
    main(args.data_dir)
