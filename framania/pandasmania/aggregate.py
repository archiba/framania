from typing import Any

import pandas
from pandas.core.groupby import GroupBy


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
