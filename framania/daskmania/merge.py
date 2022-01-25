import dask.dataframe.multi
from dask.dataframe import DataFrame
from pandas import DataFrame as PDataFrame
from typing import List, Optional
import distributed
from framania.pandasmania.merge import merge_on_columns_without_breaking_index


def merge_pandas_df_on_columns_without_breaking_index(left_df: DataFrame, right_df: PDataFrame,
                                                      on: Optional[List[str]] = None,
                                                      left_on: Optional[List[str]] = None,
                                                      right_on: Optional[List[str]] = None,
                                                      how: str = 'inner',
                                                      keep_left_index: bool = True,
                                                      keep_right_index: bool = True,
                                                      reindex_by_left: bool = True,
                                                      reindex_by_right: bool = False) \
        -> DataFrame:
    return left_df \
        .map_partitions(
        lambda df: merge_on_columns_without_breaking_index(df, right_df, on,
                                                           left_on, right_on,
                                                           how, keep_left_index, keep_right_index,
                                                           reindex_by_left, reindex_by_right))


def merge_pandas_df_future_on_columns_without_breaking_index(left_df: DataFrame, right_df: 'distributed.client.Future',
                                                             on: Optional[List[str]] = None,
                                                             left_on: Optional[List[str]] = None,
                                                             right_on: Optional[List[str]] = None,
                                                             how: str = 'inner',
                                                             keep_left_index: bool = True,
                                                             keep_right_index: bool = True,
                                                             reindex_by_left: bool = True,
                                                             reindex_by_right: bool = False) \
        -> DataFrame:
    return left_df \
        .map_partitions(
        lambda df: merge_on_columns_without_breaking_index(df, right_df.result(),
                                                           on, left_on, right_on,
                                                           how, keep_left_index, keep_right_index,
                                                           reindex_by_left, reindex_by_right))


def merge_dask_df_on_columns_without_breaking_index(left_df: DataFrame, right_df: DataFrame,
                                                    on: Optional[List[str]] = None,
                                                    left_on: Optional[List[str]] = None,
                                                    right_on: Optional[List[str]] = None,
                                                    how: str = 'inner',
                                                    keep_left_index: bool = True,
                                                    keep_right_index: bool = True,
                                                    reindex_by_left: bool = True,
                                                    reindex_by_right: bool = False):
    assert (set(left_df.divisions) <= set(right_df.divisions)) or (set(left_df.divisions) >= set(right_df.divisions))
    return dask.dataframe.multi.map_partitions(
        lambda left, right: merge_on_columns_without_breaking_index(left, right,
                                                                    on, left_on, right_on,
                                                                    how, keep_left_index, keep_right_index,
                                                                    reindex_by_left, reindex_by_right),
        left_df, right_df
    )
