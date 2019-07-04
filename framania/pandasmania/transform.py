from typing import List, Any

import numpy
import pandas
from pandas import Series, DataFrame


def timeseries_value_changed(value_series: Series, abs_threshold: float=0.0, change_at_nan: bool = True):
    """
    API to check if value of series in sorted timeseries data is changed.

    Args:
        value_series (pandas.Series): series contains some value
        abs_threshold (float): absolute threshold to check value is changed or not. default: 0.0 (any changes)
        change_at_nan (bool): if True, consider that nan means value is changed
    Returns:
        result bool pandas series
    Examples:
        >>> import pandas
        >>> pd = pandas.DataFrame({
        ...     'a': [1, 2, None, 8, 20, 23, 28, 30]
        ... })
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
             a
        0   1.0
        1   2.0
        2   NaN
        3   8.0
        4  20.0
        5  23.0
        6  28.0
        7  30.0
        >>> result = timeseries_value_changed(pd.a, 10)
        >>> print(result)
        0     True
        1    False
        2     True
        3     True
        4     True
        5    False
        6    False
        7    False
        Name: a, dtype: bool
        >>> result = timeseries_value_changed(pd.a, 10, change_at_nan=False)
        >>> print(result)
        0     True
        1    False
        2    False
        3    False
        4     True
        5    False
        6    False
        7    False
        Name: a, dtype: bool
    """


    if change_at_nan is False:
        value_series = value_series.fillna(method='ffill')

    last_value_series = value_series.shift(1)
    value_changed = ~((value_series - last_value_series).abs() < abs_threshold)
    return value_changed


def timeseries_id_changed(id_series: Series, change_at_nan: bool = True):
    """
    API to check if value of id series in sorted timeseries data is changed. (sort key must be `[id, time]`)

    Args:
        id_series (pandas.Series): series contains some id
        change_at_nan (bool): if True, consider that nan means id is changed
    Returns:
        result bool pandas series
    Examples:
        >>> import pandas
        >>> pd = pandas.DataFrame({
        ...     'a': [1, 1, 1, 1, 2, None, 3, 3]
        ... })
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
             a
        0  1.0
        1  1.0
        2  1.0
        3  1.0
        4  2.0
        5  NaN
        6  3.0
        7  3.0
        >>> result = timeseries_id_changed(pd.a)
        >>> print(result)
        0     True
        1    False
        2    False
        3    False
        4     True
        5     True
        6     True
        7    False
        Name: a, dtype: bool
        >>> result = timeseries_id_changed(pd.a, change_at_nan=False)
        >>> print(result)
        0     True
        1    False
        2    False
        3    False
        4     True
        5    False
        6     True
        7    False
        Name: a, dtype: bool
    """


    if change_at_nan is False:
        id_series = id_series.fillna(method='ffill')

    last_value_series = id_series.shift(1)

    value_changed = id_series != last_value_series
    return value_changed



def grouper_for_timeseries(split_flags: Series, start_id: int = 1):
    """
    API to build grouper id to split timeseries data by some splitting rules.

    Args:
        split_flags (pandas.Series[bool]): bool type pandas series, with True value at row where to be split
        start_id (int): start value for incremental grouper id
    Returns:
        result pandas series
    Examples:
        >>> import pandas
        >>> pd = pandas.DataFrame({
        ...     'a': [1, 2, 5, 8, 20, 23, 28, 30],
        ...     'b': [1, 1, 1, 1, 2, 2, 3, 3]
        ... })
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
            a  b
        0   1  1
        1   2  1
        2   5  1
        3   8  1
        4  20  2
        5  23  2
        6  28  3
        7  30  3
        >>> splitter = timeseries_id_changed(pd.b) | timeseries_value_changed(pd.a, 10)
        >>> print(splitter)
        0     True
        1    False
        2    False
        3    False
        4     True
        5    False
        6     True
        7    False
        dtype: bool
        >>> result = grouper_for_timeseries(splitter)
        >>> print(result)
        ... # doctest: +NORMALIZE_WHITESPACE
        0    1
        1    1
        2    1
        3    1
        4    2
        5    2
        6    3
        7    3
        dtype: int64
    """

    return split_flags.astype(int).cumsum() + (start_id - 1)


def group_row_number(pd: DataFrame, groupby: List[Any], ascending: bool=True):
    """
        API to add series that contains numbers of row in each groupby group.

        Args:
            pd (pandas.DataFrame): pandas dataframe which contains groupby columns
            groupby (List[Any]): grouping key. row number will be calculated in each of this groupby
            ascending (bool): if True, row number will starts with 0 and ends with (number of rows - 1). False means opposite
        Returns:
            result row number pandas series
        Examples:
            >>> import pandas
            >>> pd = pandas.DataFrame({
            ...     'a': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, None, 4],
            ...     'b': [100, 100, 200, 300, 50, 60, 70, 80, 10, 20, -1, 100]
            ... })
            >>> print(pd)
            ... # doctest: +NORMALIZE_WHITESPACE
                  a    b
            0   1.0  100
            1   1.0  100
            2   1.0  200
            3   1.0  300
            4   2.0   50
            5   2.0   60
            6   2.0   70
            7   2.0   80
            8   3.0   10
            9   3.0   20
            10  NaN   -1
            11  4.0  100
            >>> result = group_row_number(pd, ['a'])
            >>> print(result)
            0     0
            1     1
            2     2
            3     3
            4     0
            5     1
            6     2
            7     3
            8     0
            9     1
            10    0
            11    0
            dtype: int64
            >>> result = group_row_number(pd, ['a'], ascending=False)
            >>> print(result)
            0     3
            1     2
            2     1
            3     0
            4     3
            5     2
            6     1
            7     0
            8     1
            9     0
            10    0
            11    0
            Name: row_number, dtype: int64

        """
    temp_df = pd[groupby]
    g = temp_df.groupby(groupby, as_index=False)
    row_number = g.cumcount()
    if ascending:
        return row_number

    temp_df.loc[:, 'row_number'] = row_number
    row_count = g.count()
    m_temp_df = temp_df.merge(row_count, on=groupby, how='left')
    m_temp_df.index = temp_df.index

    m_temp_df['row_number'] = \
        m_temp_df['row_number_y'].fillna(1).astype(int) - m_temp_df['row_number_x'] - 1
    return m_temp_df['row_number']

def asstr(ps: pandas.Series, keep_nan: bool = True, remove_dotzero: bool = False):
    """
    API to make pandas Series as type string keeping NaN value.

    Args:
        ps (pandas.Series): target pandas series
        keep_nan (bool): if keep NaN as NaN or make nan as string "nan"
        remove_dotzero (bool): if remove
    Returns:
        string type pandas Series
    Examples:
        >>> import pandas
        >>> pd = pandas.DataFrame({
        ...     'a': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, None, 4.01],
        ...     'b': [100, 100, 200, 300, 50, 60, 70, 80, 10, 20, -1, 100]
        ... })
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
               a    b
        0   1.00  100
        1   1.00  100
        2   1.00  200
        3   1.00  300
        4   2.00   50
        5   2.00   60
        6   2.00   70
        7   2.00   80
        8   3.00   10
        9   3.00   20
        10   NaN   -1
        11  4.01  100
        >>> result = asstr(pd.a)
        >>> print(result)
        0      1.0
        1      1.0
        2      1.0
        3      1.0
        4      2.0
        5      2.0
        6      2.0
        7      2.0
        8      3.0
        9      3.0
        10     NaN
        11    4.01
        Name: a, dtype: object
        >>> result = asstr(pd.a, remove_dotzero=True)
        >>> print(result)
        0        1
        1        1
        2        1
        3        1
        4        2
        5        2
        6        2
        7        2
        8        3
        9        3
        10     NaN
        11    4.01
        Name: a, dtype: object
    """

    if not keep_nan:
        return ps.astype('str')

    str_s: pandas.Series = ps.astype('str')
    str_s[ps.isnull()] = numpy.nan

    if remove_dotzero:
        str_s = str_s.str.replace('.0$', '')

    return str_s
