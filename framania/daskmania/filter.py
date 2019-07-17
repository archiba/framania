from typing import List, Any, Union

from dask.dataframe import DataFrame
from framania.pandasmania.filter import \
    drop_duplicates_by_named_index_and_keys as pandas_drop_duplicates_by_named_index_and_keys


def drop_rows_by_index(dd: DataFrame, drop_index: List[Any]):
    """
    API to remove rows specified by index labels.

    Args:
        df (dask.dataframe.DataFrame): target dask dataframe
        drop_index (List[Any]): index labels to drop from dataframe
    Returns:
        result dask dataframe
    Examples:
        >>> import dask.dataframe
        >>> import pandas
        >>> pd = pandas.DataFrame({'a': range(0, 60, 10), 'b': range(0, 600, 100)}, index=range(0, 6))
        >>> print(pd)
            a    b
        0   0    0
        1  10  100
        2  20  200
        3  30  300
        4  40  400
        5  50  500
        >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
        >>> result = drop_rows_by_index(dd, [0, 2, 3])
        >>> print(result.compute())
            a    b
        1  10  100
        4  40  400
        5  50  500
    """
    return dd.map_partitions(lambda pd: pd[~pd.index.isin(drop_index)])


def drop_duplicates_by_named_index_and_keys(dd: DataFrame, subset: List[Any], keep: Union[str, bool] = 'first'):
    """
        API to do drop_duplicates with target subset including named index.

        Args:
            df (dask.dataframe.DataFrame): target dask dataframe
            subset (List[Any]): drop_duplicates subset key columns that be used with named index
            keep (Union[str, bool]): see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html
        Returns:
            result dask dataframe
        Examples:
            >>> import dask.dataframe
            >>> import pandas
            >>> pd = pandas.DataFrame({'a': [1, 10, 100, 1, 1, 100], 'b': range(0, 600, 100), 'idx': [0, 1, 2, 0, 1, 2]}).set_index('idx')
            >>> print(pd)
            ... # doctest: +NORMALIZE_WHITESPACE
                   a    b
            idx
            0      1    0
            1     10  100
            2    100  200
            0      1  300
            1      1  400
            2    100  500
            >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
            >>> result = drop_duplicates_by_named_index_and_keys(dd, ['a'])
            >>> print(result.compute())
            ... # doctest: +NORMALIZE_WHITESPACE
                   a    b
            idx
            0      1    0
            1     10  100
            1      1  400
            2    100  200
        """
    return dd.map_partitions(lambda pd: pandas_drop_duplicates_by_named_index_and_keys(pd, subset, keep))
