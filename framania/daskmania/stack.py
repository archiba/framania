from typing import Any, List

import pandas
from dask.dataframe import DataFrame
from framania.pandasmania.stack import stack_list_column as pandas_stack_list_column


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
