from typing import List, Any, Union, Tuple

from pandas import DataFrame


def drop_duplicates_by_named_index_and_keys(pd: DataFrame, subset: List[Any], keep: Union[str, bool] = 'first'):
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


def index_level_value(df: DataFrame, level_idx: Union[int, str]):
    """
    >>> df = DataFrame({'a': [1,2,3], 'b': [2,3,4], 'c': [3,4,5], 'i1': [0,0,1], 'i2': [0, 1, 0]})
    >>> df.set_index(['i1', 'i2'], inplace=True)
    >>> df
    ... # doctest: +NORMALIZE_WHITESPACE
               a  b  c
        i1 i2
        0  0   1  2  3
           1   2  3  4
        1  0   3  4  5
    >>> index_level_value(df, 0)
    Int64Index([0, 0, 1], dtype='int64', name='i1')
    >>> index_level_value(df, 'i1')
    Int64Index([0, 0, 1], dtype='int64', name='i1')
    """
    return df.index.get_level_values(level_idx)


def illoc(df: DataFrame, level_idx: Union[int, str], value: Any):
    """
    >>> df = DataFrame({'a': [1,2,3], 'b': [2,3,4], 'c': [3,4,5], 'i1': [0,0,1], 'i2': [0, 1, 0]})
    >>> df.set_index(['i1', 'i2'], inplace=True)
    >>> df
    ... # doctest: +NORMALIZE_WHITESPACE
               a  b  c
        i1 i2
        0  0   1  2  3
           1   2  3  4
        1  0   3  4  5
    >>> illoc(df, 1, 0)
    ... # doctest: +NORMALIZE_WHITESPACE
               a  b  c
        i1 i2
        0  0   1  2  3
        1  0   3  4  5
    >>> illoc(df, 'i2', 0)
    ... # doctest: +NORMALIZE_WHITESPACE
               a  b  c
        i1 i2
        0  0   1  2  3
        1  0   3  4  5
    """
    return df[df.index.get_level_values(level_idx) == value]


class ILLoc:
    def __init__(self, df: DataFrame):
        self.df = df

    def __getitem__(self, item: Tuple[Union[int, str], Any]):
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], int) or isinstance(item[0], str)
        level_index = index_level_value(self.df, item[0])
        return level_index.to_frame().loc[item[1]]


il = index_level_value
DataFrame.il = lambda self, idx: il(self, idx)
DataFrame.illoc = lambda self: ILLoc(self)
