from collections import OrderedDict
from typing import Any, List

import numpy
import pandas
from pandas import DataFrame
from pandas.core.indexes.frozen import FrozenList


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


def stack_list_columns(pd: DataFrame, list_columns: List[Any], output_dtypes: List[Any],
                       keep_columns: List[Any] = None):
    """
        API to create stack pandas dataframe from columns containing list-like objects.
        List elements which has same list index will be assigned to same row.
        If length of lists are not same, NaN will be filled automatically.

        Args:
            df (pandas.DataFrame): target pandas dataframe
            list_columns (List[Any]): name of columns containing list-like objects.
            output_dtypes (List[Any]): dtypes of output columns. may depends on contents of list.
            keep_columns (List[Any]): result dataframe will contains original index, stacked column, and keep_columns
        Returns:
            result pandas dataframe
        Examples:
            >>> import pandas
            >>> pd = pandas.DataFrame({'a1': ['1,2,3', '2,3,4', '3,4'], 'a2': ['a,b,c', 'b,c,d', 'c,d'],
            ...          'b': [1, 2, 3], 'idx': [0, 1, 2]}).set_index('idx')
            >>> pd['a1-list'] = pd['a1'].str.split(',')
            >>> pd['a2-list'] = pd['a2'].str.split(',')
            >>> print(pd)
            ... # doctest: +NORMALIZE_WHITESPACE
                    a1     a2  b    a1-list    a2-list
            idx
            0    1,2,3  a,b,c  1  [1, 2, 3]  [a, b, c]
            1    2,3,4  b,c,d  2  [2, 3, 4]  [b, c, d]
            2      3,4    c,d  3     [3, 4]     [c, d]
            >>> result = stack_list_columns(pd, ['a1-list', 'a2-list'], ['str', 'str'], ['b'])
            >>> print(result)
            ... # doctest: +NORMALIZE_WHITESPACE
                 b a1-list a2-list
            idx
            0    1       1       a
            0    1       2       b
            0    1       3       c
            1    2       2       b
            1    2       3       c
            1    2       4       d
            2    3       3       c
            2    3       4       d
            >>> pd = pandas.DataFrame({'a1': ['1,2,3', '2,3,4', '3,4'], 'a2': ['a,b,c', 'b,c', 'c,d'],
            ...          'b': [1, 2, 3], 'idx': [0, 1, 2]}).set_index('idx')
            >>> pd['a1-list'] = pd['a1'].str.split(',')
            >>> pd['a2-list'] = pd['a2'].str.split(',')
            >>> print(pd)
            ... # doctest: +NORMALIZE_WHITESPACE
                    a1     a2  b    a1-list    a2-list
            idx
            0    1,2,3  a,b,c  1  [1, 2, 3]  [a, b, c]
            1    2,3,4    b,c  2  [2, 3, 4]     [b, c]
            2      3,4    c,d  3     [3, 4]     [c, d]
            >>> result = stack_list_columns(pd, ['a1-list', 'a2-list'], ['str', 'str'], ['b'])
            >>> print(result)
            ... # doctest: +NORMALIZE_WHITESPACE
                 b a1-list a2-list
            idx
            0    1       1       a
            0    1       2       b
            0    1       3       c
            1    2       2       b
            1    2       3       c
            1    2       4     NaN
            2    3       3       c
            2    3       4       d
        """
    ipd = pd.set_index(keep_columns, append=True)

    index = []
    values = [[] for _ in list_columns]

    # TODO: more efficient stack implementation if exists
    for idx, row in ipd.iterrows():
        list_column_values = [row[list_column] for list_column in list_columns]
        max_len = max(len(lcv) for lcv in list_column_values)
        index.extend([idx] * max_len)
        for vs, _list_column_values in zip(values, list_column_values):
            vlen = len(_list_column_values)
            none_values = [numpy.nan] * (max_len - vlen)
            vs.extend(_list_column_values)
            vs.extend(none_values)
        values.extend(row)
    ix = pandas.Index(index, names=ipd.index.names)
    series_s = [pandas.Series(vs, dtype=output_dtype, index=ix) for vs, output_dtype in zip(values, output_dtypes)]
    result = pandas.DataFrame(OrderedDict([(list_column, s) for list_column, s in zip(list_columns, series_s)]))
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
    result = pandas.DataFrame(OrderedDict([(f'{dict_column}{value_suffix}', vs),
                                           (f'{dict_column}{label_suffix}', ls)]))
    result.reset_index(keep_columns, drop=False, inplace=True)
    return result


def stack_columns(pd: DataFrame, target_columns: List[Any], keep_columns: List[Any],
                  label_name: Any = 'stack_label', output_name: Any = 'stacked',
                  label_dtype: Any = 'object', output_dtype: Any = 'object'):
    """
    API to create stack pandas dataframe from specific columns.

    Args:
        df (pandas.DataFrame): target pandas dataframe
        target_columns (List[Any]): name of columns to stack.
        keep_columns (List[Any]): result dataframe will contains original index, stacked column, and keep_columns
        label_name (Any): name of label column in stack result
        output_name (Any): name of output column in stack result
        label_dtype (Any): dtype of label column in stack result
        output_dtype (Any): dtype of output column in stack result
    Returns:
        result pandas dataframe
    Examples:
        >>> import pandas
        >>> pd = pandas.DataFrame({'a': [1, 2, 3, 4], 'b': [2, 3, 4, 5], 'c': [5, 6, 7, 8],
        ...                        'label': ['a', 'b', 'c', 'd']},
        ...                        index=[100, 200, 300, 400])
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
             a  b  c label
        100  1  2  5     a
        200  2  3  6     b
        300  3  4  7     c
        400  4  5  8     d
        >>> result = stack_columns(pd, ['a', 'b'], ['label'])
        >>> print(result)
        ... # doctest: +NORMALIZE_WHITESPACE
            label stack_label  stacked
        100     a           a        1
        100     a           b        2
        200     b           a        2
        200     b           b        3
        300     c           a        3
        300     c           b        4
        400     d           a        4
        400     d           b        5
        >>> print(result.dtypes)
        ... # doctest: +NORMALIZE_WHITESPACE
        label          object
        stack_label    object
        stacked        object
        dtype: object
    """
    indexed = pd.set_index(keep_columns, append=True)
    stacked = indexed[target_columns].stack()
    new_index_names = list(stacked.index.names)
    new_index_names[-1] = label_name
    renamed = stacked.rename_axis(index=FrozenList(new_index_names))
    renamed.name = output_name

    result_df = DataFrame(renamed).reset_index(level=keep_columns + [label_name])

    result_df[label_name] = result_df[label_name].astype(label_dtype)
    result_df[output_name] = result_df[output_name].astype(output_dtype)

    return result_df
