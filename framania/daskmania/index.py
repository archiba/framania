import shutil
from base64 import b64encode
from hashlib import sha1
from pathlib import Path
from typing import Union, List, Any, Optional
from uuid import uuid4

import dask.dataframe
import pandas


def _concat_dimension_and_add_partition_key(df: pandas.DataFrame,
                                            partition_on: str, partition_value: Any,
                                            dimension_row: pandas.Series,
                                            merge_keys: Optional[List[str]] = None):
    if merge_keys is None:
        merge_keys = []
    df['__merge__'] = 1
    dimension_row = pandas.DataFrame([dimension_row])
    dimension_row['__merge__'] = 1
    merged = dimension_row.merge(df, on=merge_keys + ['__merge__'], how='left')
    del merged['__merge__']
    merged[partition_on] = partition_value
    return merged.set_index(partition_on)


def _read_parquet_then_concat_dimension_and_add_partition_key(parquet_path: Path,
                                                              partition_on: str, partition_value: Any,
                                                              dimension_row: pandas.Series,
                                                              merge_keys: Optional[List[str]] = None):
    df = pandas.read_parquet(parquet_path)
    return _concat_dimension_and_add_partition_key(df, partition_on, partition_value, dimension_row, merge_keys)


def _create_distributed_dimension_data(src_data: pandas.DataFrame, dimension_data: pandas.DataFrame,
                                       tmp_dir: Path):
    assert tmp_dir.is_dir()
    idx_name = dimension_data.index.name
    files = {}
    meta = None
    for idx, row in dimension_data.iterrows():
        dimension_concat_data = _concat_dimension_and_add_partition_key(src_data, idx_name, idx, row)
        if meta is None:
            meta = dimension_concat_data.iloc[:0]
        uid = str(uuid4())
        dimension_concat_data.to_parquet(tmp_dir / f"{uid}.parquet")
        files[idx] = uid
    return files, meta


def duplicate_pandas_for_each_distribution_key(pandas_df: pandas.DataFrame, dask_df: dask.dataframe.DataFrame,
                                               dask_dimension_columns: List[str],
                                               temporary_parquet_root: Union[str, Path],
                                               dimension_merge_keys: Optional[List[str]] = None,
                                               delete_existing_temporary_directory: bool = True):
    assert (dask_df.index.map_partitions(lambda i: i.nunique()).compute().isin([0, 1])).all(), \
        "Please reindex dask dataframe as mono-index distribution using `set_mono_index_via_disk`"
    if delete_existing_temporary_directory:
        shutil.rmtree(str(temporary_parquet_root), ignore_errors=True)
    temporary_parquet_root.mkdir(parents=True, exist_ok=True)
    dimension_df = dask_df[dask_dimension_columns].map_partitions(lambda pdf: pdf.head(1)).compute()

    assert len(dimension_df) > 0, "Dask df must not be empty."

    parquet_file = temporary_parquet_root / "pandas_df.parquet"
    pandas_df.to_parquet(parquet_file)

    keys = dimension_df.index
    name = f"framania-duplicate-pandas-for-each-distribution-key-{str(uuid4())}"
    procs = {}

    meta = None
    for i, (idx, row) in enumerate(dimension_df.iterrows()):
        procs[(name, i)] = (_read_parquet_then_concat_dimension_and_add_partition_key,
                            parquet_file,
                            dimension_df.index.name, idx, row, dimension_merge_keys)
        if i == 0:
            meta = _read_parquet_then_concat_dimension_and_add_partition_key(
                parquet_file,
                dimension_df.index.name,
                idx, row, dimension_merge_keys)

    divisions = keys + [keys[-1]]

    out_dir = temporary_parquet_root / "merge_ddf"
    dask.dataframe.DataFrame(procs, name, meta, divisions).to_parquet(out_dir)
    return dask.dataframe.read_parquet(out_dir)


def _read_parquet_files_then_concat(files: List[Union[str, Path]], partition_on: str, partition_value: Any,
                                    options: dict):
    concat_result = pandas.concat([pandas.read_parquet(f, **options) for f in files], axis=0)
    concat_result[partition_on] = partition_value
    return concat_result.set_index(partition_on)


def _read_dask_dataframe_from_partitioned_parquets(parquet_dir_root: Union[str, Path], partition_on: str, **options):
    p = Path(parquet_dir_root)
    normal_ddf = dask.dataframe.read_parquet(parquet_dir_root)
    keys = list(sorted(normal_ddf[partition_on].drop_duplicates().compute()))

    name = f"framania-read-from-partitioned-parquets-{str(uuid4())}"
    procs = {}

    divisions = keys + [keys[-1]]

    partition_dtype = normal_ddf._meta[partition_on].cat.categories.dtype
    normal_ddf._meta[partition_on] = normal_ddf._meta[partition_on].astype(partition_dtype)

    meta = normal_ddf._meta.set_index(partition_on)

    for i, key in enumerate(keys):
        files = list((p / f"{partition_on}={key}").glob("part.*.parquet"))
        procs[(name, i)] = (_read_parquet_files_then_concat, files, partition_on, key, options)

    return dask.dataframe.DataFrame(procs, name, meta, divisions)


def set_index_via_disk(df: dask.dataframe.DataFrame, column: str,
                       temporary_parquet_root: Union[str, Path],
                       delete_existing_temporary_directory: bool = True,
                       **options):
    if delete_existing_temporary_directory:
        shutil.rmtree(str(temporary_parquet_root), ignore_errors=True)
    df.to_parquet(temporary_parquet_root, partition_on=column, **options)
    idx_df = _read_dask_dataframe_from_partitioned_parquets(temporary_parquet_root,
                                                            column, **options)
    return idx_df


set_mono_index_via_disk = set_index_via_disk


def _pandas_row_to_hash(row, columns) -> int:
    return hash(tuple(row[c] for c in columns))


def set_hash_index_via_disk(df: dask.dataframe.DataFrame, columns: List[str],
                            temporary_parquet_root: Union[str, Path],
                            drop_existing_index: bool = True,
                            delete_existing_temporary_directory: bool = True,
                            **options):
    index_keys = df[columns].drop_duplicates().compute()
    index_name = f'HASH-{"-".join([str(v) for v in columns])}'
    index_keys[index_name] = index_keys.apply(_pandas_row_to_hash, columns=columns, axis=1)

    meta = pandas.DataFrame({c: df._meta[c] for c in df.columns})
    meta[index_name] = pandas.Series([], dtype='str')
    df_ = df.map_partitions(lambda df: df.reset_index(drop=drop_existing_index).merge(index_keys, on=columns),
                            meta=meta)

    if df.known_divisions:
        df_ = df_.clear_divisions()
        df_._meta.reset_index(drop=True, inplace=True)
    return set_index_via_disk(df_, index_name, temporary_parquet_root, delete_existing_temporary_directory, **options)
