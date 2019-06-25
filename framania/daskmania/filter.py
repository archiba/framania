from typing import List, Any

from dask.dataframe import DataFrame


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
