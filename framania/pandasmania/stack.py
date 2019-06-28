from typing import Any, List

import pandas
from pandas import DataFrame


def stack_list_column(pd: DataFrame, list_column: Any, output_dtype: Any, keep_columns: List[Any] = None):
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


def stack_dict_column(pd: DataFrame, dict_column: Any, label_dtype: Any, value_dtype: Any,
                      keep_columns: List[Any] = None, label_suffix: str = '_label', value_suffix: str = '_value'):
    """
    API to create stack pandas dataframe from column containing dict-like objects.

    Args:
        df (pandas.DataFrame): target pandas dataframe
        dict_column (Any): name of column containing dict-like objects.
        label_dtype (Any): dtype of keys of dict column. it depends on keys of dict.
        label_dtype (Any): dtype of values of dict column. it depends on values of dict.
        keep_columns (List[Any]): result dataframe will contains original index, stacked column, and keep_columns
        label_suffix (str): label column name of stack result dataframe will be `{dict_column}{label_suffix}`
        value_suffix (str): value column name of stack result dataframe will be `{dict_column}{value_suffix}`
    Returns:
        result pandas dataframe
    Examples:
        >>> import pandas
        >>> pd = pandas.DataFrame({'a': [{'a': 100, 'b': 200}, {'c': 300}, {'a': 500, 'c': 1000}],
        ...                        'b': [1, 2, 3], 'idx': [0, 1, 2]}).set_index('idx')
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
                                 a  b
        idx
        0     {'a': 100, 'b': 200}  1
        1               {'c': 300}  2
        2    {'a': 500, 'c': 1000}  3
        >>> result = stack_dict_column(pd, 'a', label_dtype='str', value_dtype='int', keep_columns=['b'])
        >>> print(result)
        ... # doctest: +NORMALIZE_WHITESPACE
             b  a_value a_label
        idx
        0    1      100       a
        0    1      200       b
        1    2      300       c
        2    3      500       a
        2    3     1000       c
        >>> print(result.dtypes)
        ... # doctest: +NORMALIZE_WHITESPACE
        b           int64
        a_value     int64
        a_label    object
        dtype: object
    """

    ipd = pd.set_index(keep_columns, append=True)

    index = []
    labels = []
    values = []

    # TODO: more efficient stack implementation if exists
    for idx, row in ipd[dict_column].iteritems():
        index.extend([idx] * len(row))
        for l, v in row.items():
            values.append(v)
            labels.append(l)
    ix = pandas.Index(index, names=ipd.index.names)
    ls = pandas.Series(labels, dtype=label_dtype, index=ix)
    vs = pandas.Series(values, dtype=value_dtype, index=ix)
    result = pandas.DataFrame({f'{dict_column}{value_suffix}': vs,
                               f'{dict_column}{label_suffix}': ls})
    result.reset_index(keep_columns, drop=False, inplace=True)
    return result
