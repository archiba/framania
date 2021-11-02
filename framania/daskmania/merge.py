from dask.dataframe import DataFrame
from pandas import DataFrame as PDataFrame
from typing import List
from framania.pandasmania.merge import merge_on_columns_without_breaking_index


def merge_pandas_df_on_columns_without_breaking_index(left_df: DataFrame, right_df: PDataFrame,
                                                      left_on: List[str], right_on: List[str],
                                                      how: str,
                                                      keep_left_index: bool = True,
                                                      keep_right_index: bool = True,
                                                      reindex_by_left: bool = True,
                                                      reindex_by_right: bool = False) \
        -> DataFrame:
    return left_df \
        .map_partitions(
        lambda df: merge_on_columns_without_breaking_index(df, right_df,
                                                           left_on, right_on,
                                                           how, keep_left_index, keep_right_index,
                                                           reindex_by_left, reindex_by_right))


def merge_pandas_df_future_on_columns_without_breaking_index(left_df: DataFrame, right_df: 'distributed.client.Future',
                                                      left_on: List[str], right_on: List[str],
                                                      how: str,
                                                      keep_left_index: bool = True,
                                                      keep_right_index: bool = True,
                                                      reindex_by_left: bool = True,
                                                      reindex_by_right: bool = False) \
        -> DataFrame:
    return left_df \
        .map_partitions(
        lambda df: merge_on_columns_without_breaking_index(df, right_df.result(),
                                                           left_on, right_on,
                                                           how, keep_left_index, keep_right_index,
                                                           reindex_by_left, reindex_by_right))