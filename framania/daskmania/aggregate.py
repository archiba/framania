from typing import List, Dict, Union, Any

from dask.dataframe import DataFrame
from pandas import MultiIndex

from framania.pandasmania.aggregate import aggregate_as_series as aggregate_as_series_pandas


def flatten_aggregated_columns(dd: DataFrame):
    """
    API to make aggregated columns that are MultiIndex style flat

    Args:
        pd: target dataframe
    Returns:
        result pandas dataframe
    Examples:
        >>> import dask.dataframe
        >>> import pandas
        >>> pd = pandas.DataFrame({'a': [1, 10, 100, 1, 1, 100], 'b': range(0, 600, 100), 'key': [0, 1, 2, 0, 1, 2]})
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
             a    b  key
        0    1    0    0
        1   10  100    1
        2  100  200    2
        3    1  300    0
        4    1  400    1
        5  100  500    2
        >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
        >>> groupby = dd.groupby(['key'])
        >>> groupby_result = groupby.agg({'a': ['sum', 'min'], 'b': ['mean', 'sum', 'max']})
        >>> print(groupby_result.compute())
        ... # doctest: +NORMALIZE_WHITESPACE
               a           b
             sum  min   mean  sum  max
        key
        0      2    1  150.0  300  300
        1     11    1  250.0  500  400
        2    200  100  350.0  700  500
        >>> flatten = flatten_aggregated_columns(groupby_result)
        >>> print(flatten.compute())
        ... # doctest: +NORMALIZE_WHITESPACE
             a_sum  a_min  b_mean  b_sum  b_max
        key
        0        2      1   150.0    300    300
        1       11      1   250.0    500    400
        2      200    100   350.0    700    500
        >>> print(flatten_aggregated_columns(dd).compute())
        ... # doctest: +NORMALIZE_WHITESPACE
             a    b  key
        0    1    0    0
        1   10  100    1
        2  100  200    2
        3    1  300    0
        4    1  400    1
        5  100  500    2
    """
    if not isinstance(dd.columns, MultiIndex) or dd.columns.nlevels != 2:
        return dd

    result = dd.copy()

    columns = []
    for l1, l2 in zip(dd.columns.get_level_values(0), dd.columns.get_level_values(1)):
        if l2 == '':
            columns.append(l1)
        else:
            columns.append('_'.join((l1, l2)))
    result.columns = columns

    return result


def aggregate_by_named_index_and_keys(dd: DataFrame, keys: List[Any], agg: Dict[str, Union[str, List[str]]]):
    """
    API for groupby aggregation by named index and other columns.

    Args:
        df (dask.dataframe.DataFrame): target dask dataframe
        keys (List[Any]): groupby key columns that be used with named index
        agg (Dict[str, Union[str, List[str]]]): see `func` in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html
    Returns:
        result dask dataframe
    Examples:
        >>> import dask.dataframe
        >>> import pandas
        >>> pd = pandas.DataFrame({'a': [1, 10, 100, 1, 1, 100], 'b': range(0, 600, 100), 'idx': [0, 1, 2, 0, 1, 2]}).set_index('idx')
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
               a    b
        idx
        0      1    0
        1     10  100
        2    100  200
        0      1  300
        1      1  400
        2    100  500
        >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
        >>> result = aggregate_by_named_index_and_keys(dd, ['a'], {'b': ['mean', 'max']})
        >>> print(result.compute())
        ... # doctest: +NORMALIZE_WHITESPACE
               a      b
                   mean  max
        idx
        0      1  150.0  300
        1      1  400.0  400
        1     10  100.0  100
        2    100  350.0  500
    """
    index_name = dd.index.name
    groupby_keys = [index_name] + keys

    lambda_agg = lambda pd: pd.groupby(by=groupby_keys).agg(agg)
    lambda_reset_index = lambda pd: pd.reset_index(keys)

    if len(keys) > 0:
        lambda_all = lambda pd: lambda_reset_index(lambda_agg(pd))
    else:
        lambda_all = lambda_agg
    return dd.map_partitions(lambda_all)


def aggregate_as_series_by_named_index_and_keys(dd: DataFrame, keys: List[Any], series: Any, agg_func: Any,
                                                output_series_name: Any):
    """
    API to output the aggregate result by named index and other columns as a column of the original data frame.

    Args:
        dd (DataFrame): target dask dataframe
        keys (List[Any): group keys other than index
        series (Any): name of series to be aggregated
        agg_func (Union[str, List[str]]): see `func` in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.agg.html
        output_series_name (Any): name of result series
    Returns:
        result pandas series
    Examples:
        >>> import dask.dataframe
        >>> import pandas
        >>> pd = pandas.DataFrame({'a': [1, 10, 100, 1, 1, 100], 'b': range(0, 600, 100), 'idx': [0, 1, 2, 0, 1, 2]}).set_index('idx')
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
               a    b
        idx
        0      1    0
        1     10  100
        2    100  200
        0      1  300
        1      1  400
        2    100  500
        >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
        >>> result = aggregate_as_series_by_named_index_and_keys(dd, ['a'], 'b', 'sum', 'sum_b')
        >>> print(result.compute())
        ... # doctest: +NORMALIZE_WHITESPACE
               a    b  sum_b
        idx
        0      1    0    300
        0      1  300    300
        1     10  100    100
        1      1  400    400
        2    100  200    700
        2    100  500    700
    """
    index_name = dd.index.name
    groupby_keys = [index_name] + keys

    def assign_result_column(pd):
        pd.reset_index(drop=False, inplace=True)
        agg_result = aggregate_as_series_pandas(pd.groupby(groupby_keys), series, agg_func)
        pd[output_series_name] = agg_result
        pd.set_index(index_name, drop=True, inplace=True)
        return pd

    return dd.map_partitions(assign_result_column)


def aggregate_as_series(dd: DataFrame, keys: List[Any], series: Any, agg_func: Any,
                        output_series_name: Any):
    """
    API to output the aggregate result as a column of the original data frame.

    Args:
        dd (DataFrame): target dask dataframe
        keys (List[Any): group keys
        series (Any): name of series to be aggregated
        agg_func (Union[str, List[str]]): see `func` in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.agg.html
        output_series_name (Any): name of result series
    Returns:
        result pandas series
    Examples:
        >>> import dask.dataframe
        >>> import pandas
        >>> pd = pandas.DataFrame({'a': [1, 10, 100, 1, 1, 100], 'b': range(0, 600, 100), 'idx': [0, 1, 2, 0, 1, 2]})
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
             a    b  idx
        0    1    0    0
        1   10  100    1
        2  100  200    2
        3    1  300    0
        4    1  400    1
        5  100  500    2
        >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
        >>> result = aggregate_as_series(dd, ['idx', 'a'], 'b', 'sum', 'sum_b')
        >>> print(result.compute())
        ... # doctest: +NORMALIZE_WHITESPACE
             a    b  idx  sum_b
        0    1    0    0    300
        1   10  100    1    100
        2  100  200    2    700
        0    1  300    0    300
        1    1  400    1    400
        2  100  500    2    700
    """
    group_data = dd.groupby(keys)[series].agg(agg_func)
    group_data.name = output_series_name
    group_df = group_data.reset_index()
    return dd.merge(group_df, on=keys)
