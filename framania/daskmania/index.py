from pathlib import Path
from typing import Union, List, Any
from uuid import uuid4

import dask.dataframe
import pandas


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
                       **options):
    df.to_parquet(temporary_parquet_root, partition_on=column, **options)
    idx_df = _read_dask_dataframe_from_partitioned_parquets(temporary_parquet_root,
                                                            column, **options)
    return idx_df
