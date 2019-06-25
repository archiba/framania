from typing import Any, List

import pandas
from pandas import DataFrame


def stack_list_column(pd: DataFrame, list_column: Any, output_dtype: Any, keep_columns: List[Any]=None):
    """
    API to create stack pandas dataframe from column containing list-like objects.

    Args:
        df (pandas.DataFrame): target pandas dataframe
        list_column (Any): name of column containing list-like objects.
        output_dtype (Any): dtype of output column. may depends on contents of list.
        keep_columns (List[Any]): result dataframe will contains original index, stacked column, and keep_columns
    Returns:
        result pandas dataframe
    Examples:
        >>> import pandas
        >>> pd = pandas.DataFrame({'a': ['1,2,3', '2,3,4', '3,4'], 'b': [1, 2, 3], 'idx': [0, 1, 2]}).set_index('idx')
        >>> pd['a-list'] = pd['a'].str.split(',')
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
                 a  b     a-list
        idx
        0    1,2,3  1  [1, 2, 3]
        1    2,3,4  2  [2, 3, 4]
        2      3,4  3     [3, 4]
        >>> result = stack_list_column(pd, 'a-list', 'str', ['b'])
        >>> print(result)
        ... # doctest: +NORMALIZE_WHITESPACE
             b a-list
        idx
        0    1      1
        0    1      2
        0    1      3
        1    2      2
        1    2      3
        1    2      4
        2    3      3
        2    3      4
    """

    ipd = pd.set_index(keep_columns, append=True)

    index = []
    values = []

    # TODO: more efficient stack implementation if exists
    for idx, row in ipd[list_column].iteritems():
        index.extend([idx] * len(row))
        values.extend(row)
    ix = pandas.Index(index, names=ipd.index.names)
    s = pandas.Series(values, dtype=output_dtype, index=ix)
    result = pandas.DataFrame({list_column: s})
    result.reset_index(keep_columns, drop=False, inplace=True)
    return result
