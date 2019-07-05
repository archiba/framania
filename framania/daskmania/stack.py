from collections import OrderedDict
from typing import Any, List

import pandas
from dask.dataframe import DataFrame

from framania.daskmania.meta import make_meta, map_partitions_as_meta
from framania.pandasmania.stack import stack_list_column as pandas_stack_list_column
from framania.pandasmania.stack import stack_list_columns as pandas_stack_list_columns
from framania.pandasmania.stack import stack_dict_column as pandas_stack_dict_column
from framania.pandasmania.stack import stack_columns as pandas_stack_columns


def stack_list_column(dd: DataFrame, list_column: Any, output_dtype: Any, keep_columns: List[Any] = None):
    """
    API to create stack dask dataframe from column containing list-like objects.

    Args:
        df (dask.dataframe.DataFrame): target dask dataframe
        list_column (Any): name of column containing list-like objects.
        output_dtype (Any): dtype of output column. may depends on contents of list.
        keep_columns (List[Any]): result dataframe will contains original index, stacked column, and keep_columns
    Returns:
        result dask dataframe
    Examples:
        >>> import dask.dataframe
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
        >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
        >>> result = stack_list_column(dd, 'a-list', 'str', ['b'])
        >>> print(result.compute())
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
    meta = make_meta((dd.index.name, dd.index.dtype),
                     [(k_column, dd[k_column].dtype) for k_column in keep_columns] +
                     [(list_column, output_dtype)])
    return dd.map_partitions(lambda pd: pandas_stack_list_column(pd, list_column, output_dtype, keep_columns),
                             meta=meta)


def stack_list_columns(dd: DataFrame, list_columns: List[Any], output_dtypes: List[Any],
                       keep_columns: List[Any] = None):
    """
        API to create stack dask dataframe from columns containing list-like objects.
        List elements which has same list index will be assigned to same row.
        If length of lists are not same, NaN will be filled automatically.

        Args:
            df (pandas.DataFrame): target dask dataframe
            list_columns (List[Any]): name of columns containing list-like objects.
            output_dtypes (List[Any]): dtypes of output columns. may depends on contents of list.
            keep_columns (List[Any]): result dataframe will contains original index, stacked column, and keep_columns
        Returns:
            result dask dataframe
        Examples:
            >>> import dask.dataframe
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
            >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
            >>> result = stack_list_columns(dd, ['a1-list', 'a2-list'], ['str', 'str'], ['b'])
            >>> print(result.compute())
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
            >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
            >>> result = stack_list_columns(dd, ['a1-list', 'a2-list'], ['str', 'str'], ['b'])
            >>> print(result.compute())
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
    meta = make_meta((dd.index.name, dd.index.dtype),
                     [(k_column, dd[k_column].dtype) for k_column in keep_columns] +
                     [(l_column, l_column_t) for l_column, l_column_t in zip(list_columns, output_dtypes)])
    return map_partitions_as_meta(dd,
                                  lambda pd: pandas_stack_list_columns(pd, list_columns, output_dtypes, keep_columns),
                                  meta)

def stack_dict_column(dd: DataFrame, dict_column: Any, label_dtype: Any, value_dtype: Any,
                      keep_columns: List[Any] = None, label_suffix: str = '_label', value_suffix: str = '_value'):
    """
    API to create stack pandas dataframe from column containing dict-like objects.

    Args:
        df (pandas.DataFrame): target dask dataframe
        dict_column (Any): name of column containing dict-like objects.
        label_dtype (Any): dtype of keys of dict column. it depends on keys of dict.
        label_dtype (Any): dtype of values of dict column. it depends on values of dict.
        keep_columns (List[Any]): result dataframe will contains original index, stacked column, and keep_columns
        label_suffix (str): label column name of stack result dataframe will be `{dict_column}{label_suffix}`
        value_suffix (str): value column name of stack result dataframe will be `{dict_column}{value_suffix}`
    Returns:
        result pandas dataframe
    Examples:
        >>> import dask.dataframe
        >>> import pandas
        >>> pd = pandas.DataFrame({'a': [{'a': 100, 'b': 200}, {'c': 300}, {'a': 500, 'c': 1000}],
        ...                        'b': [1, 2, 3], 'idx': ['A', 'B', 'C']}).set_index('idx')
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
                                 a  b
        idx
        A     {'a': 100, 'b': 200}  1
        B               {'c': 300}  2
        C    {'a': 500, 'c': 1000}  3
        >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
        >>> result = stack_dict_column(dd, 'a', label_dtype='str', value_dtype='int', keep_columns=['b'])
        >>> print(result.compute())
        ... # doctest: +NORMALIZE_WHITESPACE
             b  a_value a_label
        idx
        A    1      100       a
        A    1      200       b
        B    2      300       c
        C    3      500       a
        C    3     1000       c
    """
    meta = make_meta((dd.index.name, dd.index.dtype),
                     [(k_column, dd[k_column].dtype) for k_column in keep_columns] +
                     [(f'{dict_column}{value_suffix}', value_dtype), (f'{dict_column}{label_suffix}', label_dtype)])
    return dd.map_partitions(
        lambda pd: pandas_stack_dict_column(pd, dict_column,
                                            label_dtype, value_dtype, keep_columns,
                                            label_suffix, value_suffix),
        meta=meta,
    )


def stack_columns(dd: DataFrame, target_columns: List[Any], keep_columns: List[Any],
                  label_name: Any = 'stack_label', output_name: Any = 'stacked',
                  label_dtype: Any = 'object', output_dtype: Any = 'object'):
    """
    API to create stack dask dataframe from specific columns.

    Args:
        df (pandas.DataFrame): target pandas dataframe
        target_columns (List[Any]): name of columns to stack.
        keep_columns (List[Any]): result dataframe will contains original index, stacked column, and keep_columns
        label_name (Any): name of label column in stack result
        output_name (Any): name of output column in stack result
        label_dtype (Any): dtype of label column in stack result
        output_dtype (Any): dtype of output column in stack result
    Returns:
        result dask dataframe
    Examples:
        >>> import dask.dataframe
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
        >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
        >>> result = stack_columns(dd, ['a', 'b'], ['label'], label_dtype='object', output_dtype='int')
        >>> print(result.compute())
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
        stacked         int64
        dtype: object
    """
    meta = make_meta((dd.index.name, dd.index.dtype),
                     [(k_column, dd[k_column].dtype) for k_column in keep_columns] +
                     [(label_name, label_dtype), (output_name, output_dtype)])
    return dd.map_partitions(lambda pd: pandas_stack_columns(pd, target_columns, keep_columns,
                                                             label_name, output_name, label_dtype, output_dtype),
                             meta=meta)
