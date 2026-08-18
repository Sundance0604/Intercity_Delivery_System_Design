"""Build and validate an indexed SQLite store from the large CFS PUMS CSV."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from contextlib import closing
from typing import Optional, Sequence

import pandas as pd

from intercity_delivery.data.cfs_processor import (
    CODE_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
)


TABLE_NAME = "shipments"
SCHEMA_VERSION = 1
SQLITE_SUFFIXES = {".sqlite", ".sqlite3", ".db"}

SQL_TYPES = {
    **{column: "TEXT" for column in CODE_COLUMNS},
    **{column: "REAL" for column in NUMERIC_COLUMNS},
}


def is_sqlite_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SQLITE_SUFFIXES


def _normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk.columns = [str(column).strip().upper() for column in chunk.columns]
    missing = set(REQUIRED_COLUMNS) - set(chunk.columns)
    if missing:
        raise ValueError(f"CSV 缺少 CFS 必需字段：{', '.join(sorted(missing))}")
    result = chunk.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in CODE_COLUMNS:
        result[column] = result[column].astype("string").str.strip()
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _create_schema(connection: sqlite3.Connection) -> None:
    columns = ", ".join(
        f'"{column}" {SQL_TYPES[column]}' for column in REQUIRED_COLUMNS
    )
    connection.execute(f'CREATE TABLE "{TABLE_NAME}" ({columns})')
    connection.execute(
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def _create_indexes(connection: sqlite3.Connection) -> None:
    connection.execute(
        f'CREATE INDEX idx_{TABLE_NAME}_od '
        f'ON "{TABLE_NAME}" (ORIG_CFS_AREA, DEST_CFS_AREA)'
    )
    connection.execute(
        f'CREATE INDEX idx_{TABLE_NAME}_od_mode '
        f'ON "{TABLE_NAME}" (ORIG_CFS_AREA, DEST_CFS_AREA, MODE)'
    )
    connection.execute(
        f'CREATE INDEX idx_{TABLE_NAME}_filter '
        f'ON "{TABLE_NAME}" (MODE, EXPORT_YN, SHIPMT_DIST_GC)'
    )


def validate_sqlite_store(path: str | Path) -> dict:
    """Validate the schema and return stored build metadata."""

    database = Path(path)
    if not database.is_file():
        raise FileNotFoundError(f"找不到 SQLite 文件：{database}")
    with closing(sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (TABLE_NAME,),
        ).fetchone()
        if table is None:
            raise ValueError(f"SQLite 中不存在表 {TABLE_NAME!r}。")
        columns = {
            row[1] for row in connection.execute(f'PRAGMA table_info("{TABLE_NAME}")')
        }
        missing = set(REQUIRED_COLUMNS) - columns
        if missing:
            raise ValueError(
                f"SQLite shipments 表缺少字段：{', '.join(sorted(missing))}"
            )
        metadata_exists = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'"
        ).fetchone()
        metadata: dict = {}
        if metadata_exists:
            for key, value in connection.execute("SELECT key, value FROM metadata"):
                try:
                    metadata[key] = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    metadata[key] = value
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
                (TABLE_NAME,),
            )
        }
        required_indexes = {"idx_shipments_od", "idx_shipments_od_mode", "idx_shipments_filter"}
        if not required_indexes.issubset(indexes):
            raise ValueError("SQLite 缺少查询索引：" + ", ".join(sorted(required_indexes - indexes)))
        metadata["indexes"] = sorted(indexes)
        metadata["row_count"] = int(
            connection.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0]
        )
        metadata["database_path"] = str(database.resolve())
        return metadata


def build_sqlite_store(
    input_path: str | Path,
    output_path: str | Path,
    chunksize: int = 250_000,
    overwrite: bool = False,
) -> dict:
    """Stream a CFS CSV/ZIP/GZ into an indexed SQLite database.

    The database is written to a sibling ``.building`` file and atomically
    renamed only after data, indexes, and metadata have been committed.
    """

    source = Path(input_path)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到 CFS 输入文件：{source}")
    if chunksize <= 0:
        raise ValueError("chunksize 必须大于 0。")
    if not is_sqlite_path(output):
        raise ValueError("输出扩展名必须为 .sqlite、.sqlite3 或 .db。")
    if output.exists() and not overwrite:
        raise FileExistsError(f"SQLite 已存在：{output}；如需重建请使用 --overwrite。")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.building")
    if temporary.exists():
        temporary.unlink()

    started = time.time()
    row_count = 0
    chunk_count = 0
    try:
        with closing(sqlite3.connect(temporary)) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA cache_size=-131072")
            _create_schema(connection)

            try:
                reader = pd.read_csv(
                    source,
                    usecols=list(REQUIRED_COLUMNS),
                    dtype=CODE_COLUMNS,
                    chunksize=chunksize,
                    low_memory=False,
                    compression="infer",
                )
            except ValueError as exc:
                raise ValueError(
                    "输入文件缺少 2022 CFS PUMS 必需字段；请确认传入官方 CSV。"
                ) from exc

            for chunk_count, chunk in enumerate(reader, start=1):
                normalized = _normalize_chunk(chunk)
                normalized.to_sql(
                    TABLE_NAME,
                    connection,
                    if_exists="append",
                    index=False,
                )
                row_count += len(normalized)
                connection.commit()
                print(
                    f"[SQLite] chunk={chunk_count}, rows={row_count:,}, "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

            print("[SQLite] 数据导入完成，正在创建索引……", flush=True)
            _create_indexes(connection)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "source_path": str(source.resolve()),
                "source_size_bytes": source.stat().st_size,
                "source_mtime_ns": source.stat().st_mtime_ns,
                "row_count": row_count,
                "chunk_count": chunk_count,
                "required_columns": list(REQUIRED_COLUMNS),
                "created_at_unix": time.time(),
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [(key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
            )
            connection.commit()

        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    result = validate_sqlite_store(output)
    result["elapsed_seconds"] = round(time.time() - started, 2)
    result["database_size_bytes"] = output.stat().st_size
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分块把大型 2022 CFS PUMS CSV 转换为带索引的 SQLite。"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只检查现有 --output 的结构和行数，不重新导入。",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = parse_args(argv)
    if args.validate_only:
        result = validate_sqlite_store(args.output)
    else:
        result = build_sqlite_store(
            args.input,
            args.output,
            chunksize=args.chunksize,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
