from typing import List, Any, Union

from pandas import DataFrame


def drop_duplicates_by_named_index_and_keys(pd: DataFrame, subset: List[Any], keep: Union[str, bool]='first'):
    """
        API to do drop_duplicates with target subset including named index.

        Args:
            df (dask.dataframe.DataFrame): target dask dataframe
            subset (List[Any]): drop_duplicates subset key columns that be used with named index
            keep (Union[str, bool]): see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html
        Returns:
            result dask dataframe
        Examples:
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
            >>> result = drop_duplicates_by_named_index_and_keys(pd, ['a'])
            >>> print(result)
            ... # doctest: +NORMALIZE_WHITESPACE
                   a    b
            idx
            0      1    0
            1     10  100
            1      1  400
            2    100  200
            >>> pd = pandas.DataFrame({'a': [1, 10, 100, 1, 1, 100],
            ...                        'b': range(0, 600, 100),
            ...                        'idx1': [0, 1, 2, 0, 1, 2],
            ...                        'idx2': [1, 1, 1, 1, 1, 1]}).set_index(['idx1', 'idx2'])
            >>> print(pd)
            ... # doctest: +NORMALIZE_WHITESPACE
                         a    b
            idx1 idx2
            0    1       1    0
            1    1      10  100
            2    1     100  200
            0    1       1  300
            1    1       1  400
            2    1     100  500
            >>> result = drop_duplicates_by_named_index_and_keys(pd, ['a'])
            >>> print(result)
            ... # doctest: +NORMALIZE_WHITESPACE
                         a    b
            idx1 idx2
            0    1       1    0
            1    1      10  100
                 1       1  400
            2    1     100  200
        """
    index_all_levels = [l for l in range(pd.index.nlevels)]
    return pd.groupby(level=index_all_levels, group_keys=False).apply(lambda pd: pd.drop_duplicates(subset=subset))
