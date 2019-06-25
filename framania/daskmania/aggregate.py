from typing import List, Dict, Union, Any

from dask.dataframe import DataFrame


def aggregate_by_named_index_and_keys(dd: DataFrame, keys: List[Any], agg: Dict[str, Union[str, List[str]]]):
    """
    API for groupby aggregation by named index and other columns.

    Args:
        df (dask.dataframe.DataFrame): target dask dataframe
        keys (List[Any]): groupby key columns that be used with named index
        agg (Dict[str, Union[str, List[str]]]): see `func` in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html
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
        >>> result = aggregate_by_named_index_and_keys(dd, ['a'], {'b': ['mean', 'max']})
        >>> print(result.compute())
        ... # doctest: +NORMALIZE_WHITESPACE
               a    b
                 mean  max
        idx
        0      1  150  300
        1      1  400  400
        1     10  100  100
        2    100  350  500
    """
    index_name = dd.index.name
    groupby_keys = [index_name] + keys

    lambda_agg = lambda pd: pd.groupby(by=groupby_keys).agg(agg)
    lambda_reset_index = lambda pd: pd.reset_index(keys)

    if len(keys) > 0:
        lambda_all = lambda pd: lambda_reset_index(lambda_agg(pd))
    else:
        lambda_all = lambda_agg
    return dd.map_partitions(lambda_all)
