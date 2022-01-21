from typing import Any

import pandas
from pandas import DataFrame, MultiIndex
from pandas.core.groupby import GroupBy


def flatten_aggregated_columns(pd: DataFrame):
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
        >>> groupby = pd.groupby(['key'])
        >>> groupby_result = groupby.agg({'a': ['sum', 'min'], 'b': ['mean', 'sum', 'max']})
        >>> print(groupby_result)
        ... # doctest: +NORMALIZE_WHITESPACE
               a           b
             sum  min   mean  sum  max
        key
        0      2    1  150.0  300  300
        1     11    1  250.0  500  400
        2    200  100  350.0  700  500
        >>> flatten = flatten_aggregated_columns(groupby_result)
        >>> print(flatten)
        ... # doctest: +NORMALIZE_WHITESPACE
             a_sum  a_min  b_mean  b_sum  b_max
        key
        0        2      1   150.0    300    300
        1       11      1   250.0    500    400
        2      200    100   350.0    700    500
        >>> print(flatten_aggregated_columns(pd))
        ... # doctest: +NORMALIZE_WHITESPACE
             a    b  key
        0    1    0    0
        1   10  100    1
        2  100  200    2
        3    1  300    0
        4    1  400    1
        5  100  500    2
    """
    if not isinstance(pd.columns, MultiIndex) or pd.columns.nlevels != 2:
        return pd

    columns = []
    for l1, l2 in zip(pd.columns.get_level_values(0), pd.columns.get_level_values(1)):
        if l2 == '':
            columns.append(l1)
        else:
            columns.append('_'.join((l1, l2)))
    result = pd.set_axis(columns, axis=1, inplace=False)
    return result

def aggregate_as_series(groupby: GroupBy, series: Any, agg_func: Any):
    """
    API to output the aggregate result as a column of the original data frame.

    Args:
        groupby (GroupBy): result of pandas.DataFrame.groupby
        series (Any): name of series to be aggregated
        agg_func (Union[str, List[str]]): see `func` in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.agg.html
    Returns:
        result pandas series
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
        >>> groupby = pd.groupby(['key', 'a'])
        >>> pd['sum_b'] = aggregate_as_series(groupby, 'b', 'sum')
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
             a    b  key  sum_b
        0    1    0    0    300
        1   10  100    1    100
        2  100  200    2    700
        3    1  300    0    300
        4    1  400    1    400
        5  100  500    2    700
    """
    def _series_agg(s, agg_func):
        new_s = pandas.Series(s.agg(agg_func), index=s.index)
        return new_s

    return groupby[series].apply(_series_agg, agg_func=agg_func)
