import pandas
from framania.pandasmania.transform import grouper_for_timeseries, group_row_number


def fbfill_series(ps: pandas.Series, limit=None):
    """
        API to fill na values by closer one of last value or next value. (closer means one of row position is close.)
        NOTICE Position of row will NOT be sorted by index automatically. Original order of rows is kept.

        Args:
            ps (pandas.Series): target pandas dataframe
            limit (optional int): see https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.fillna.html#pandas.DataFrame.fillna
        Returns:
            new pandas series with filled values
        Examples:
            >>> import pandas
            >>> pd = pandas.DataFrame({'a': [1, 2, 3, None, None, None, 4, 5, None, 6, None, None, 7, None, 8]},
            ...                       index=[0, 14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7])
            >>> print(pd)
            ... # doctest: +NORMALIZE_WHITESPACE
                  a
            0   1.0
            14  2.0
            1   3.0
            13  NaN
            2   NaN
            12  NaN
            3   4.0
            11  5.0
            4   NaN
            10  6.0
            5   NaN
            9   NaN
            6   7.0
            8   NaN
            7   8.0
            >>> result = fbfill_series(pd.a)
            >>> print(result)
            ... # doctest: +NORMALIZE_WHITESPACE
            0     1.0
            14    2.0
            1     3.0
            13    3.0
            2     3.0
            12    4.0
            3     4.0
            11    5.0
            4     5.0
            10    6.0
            5     6.0
            9     7.0
            6     7.0
            8     7.0
            7     8.0
            Name: a, dtype: float64
        """
    ffill_series = ps.fillna(method='ffill', axis=0, inplace=False, limit=limit)
    bfill_series = ps.fillna(method='bfill', axis=0, inplace=False, limit=limit)

    nan_value_flags = ps.isnull() & ~ps.shift(1).isnull()
    temp_df = pandas.DataFrame({'v': ps, 'na_flag': nan_value_flags}, index=ps.index)
    temp_df['nan_group'] = grouper_for_timeseries(temp_df.na_flag)
    temp_df = temp_df[ps.isnull()]
    temp_df['fnrow'] = group_row_number(temp_df, ['nan_group'], True)
    temp_df['bnrow'] = group_row_number(temp_df, ['nan_group'], False)

    ffill_index = ps.isnull() & (temp_df.fnrow <= temp_df.bnrow)
    rfill_index = ps.isnull() & (temp_df.fnrow > temp_df.bnrow)

    result = pandas.Series(ps)
    result.loc[ffill_index] = ffill_series.loc[ffill_index]
    result.loc[rfill_index] = bfill_series.loc[rfill_index]
    return result
