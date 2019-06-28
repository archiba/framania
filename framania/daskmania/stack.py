from typing import Any, List

import pandas
from dask.dataframe import DataFrame
from framania.pandasmania.stack import stack_list_column as pandas_stack_list_column
from framania.pandasmania.stack import stack_dict_column as pandas_stack_dict_column


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
    meta = {k_column: dd[k_column].dtype for k_column in keep_columns}
    meta[list_column] = output_dtype

    return dd.map_partitions(lambda pd: pandas_stack_list_column(pd, list_column, output_dtype, keep_columns),
                             meta=meta)


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
    meta = {k_column: dd._meta[k_column] for k_column in keep_columns}
    meta[f'{dict_column}{value_suffix}'] = pandas.Series([], dtype=value_dtype)
    meta[f'{dict_column}{label_suffix}'] = pandas.Series([], dtype=label_dtype)
    meta = pandas.DataFrame(meta, index=dd._meta.index)

    return dd.map_partitions(
        lambda pd: pandas_stack_dict_column(pd, dict_column,
                                            label_dtype, value_dtype, keep_columns,
                                            label_suffix, value_suffix),
        meta=meta,
        )
