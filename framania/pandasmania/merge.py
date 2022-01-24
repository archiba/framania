from typing import List, Optional
from pandas import DataFrame, MultiIndex


def merge_on_columns_without_breaking_index(left_df: DataFrame, right_df: DataFrame,
                                            on: Optional[List[str]]=None,
                                            left_on: Optional[List[str]]=None, right_on: Optional[List[str]]=None,
                                            how: str = 'inner',
                                            keep_left_index: bool = True,
                                            keep_right_index: bool = True,
                                            reindex_by_left: bool = True,
                                            reindex_by_right: bool = False) \
        -> DataFrame:
    """
    >>> df1 = DataFrame({"A": [1,2,3,4,5,1,2,3,4,5],
    ...                  "B": [2,2,3,3,4,4,5,5,6,6],
    ...                  "C": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']}).set_index('A')
    >>> df2 = DataFrame({"a": [1,2,3,4,5,1,2,3,4,5],
    ...                  "b": [2,2,3,3,4,4,5,5,6,6],
    ...                  "c": ['aa', 'ba', 'ca', 'da', 'ea', 'fa', 'ga', 'ha', 'ia', 'ja']}).set_index('a')
    >>> result = merge_on_columns_without_breaking_index(df1, df2, left_on=["A", "B"], right_on=["a", "b"], how="inner")
    >>> result
    ... # doctest: +NORMALIZE_WHITESPACE
           B  C  a  b   c
        A
        1  2  a  1  2  aa
        2  2  b  2  2  ba
        3  3  c  3  3  ca
        4  3  d  4  3  da
        5  4  e  5  4  ea
        1  4  f  1  4  fa
        2  5  g  2  5  ga
        3  5  h  3  5  ha
        4  6  i  4  6  ia
        5  6  j  5  6  ja
    >>> df1 = DataFrame({"A": [1,2,3,4,5,1,2,3,4,5],
    ...                  "B": [2,2,3,3,4,4,5,5,6,6],
    ...                  "C": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']}).set_index(["A", "B"])
    >>> df2 = DataFrame({"a": [1,2,3,4,5,1,2,3,4,5],
    ...                  "b": [2,2,3,3,4,4,5,5,6,6],
    ...                  "c": ['aa', 'ba', 'ca', 'da', 'ea', 'fa', 'ga', 'ha', 'ia', 'ja']}).set_index(['a', 'b'])
    >>> result = merge_on_columns_without_breaking_index(df1, df2, left_on=["A", "B"], right_on=["a", "b"], how="inner")
    >>> result
    ... # doctest: +NORMALIZE_WHITESPACE
             C  a  b   c
        A B
        1 2  a  1  2  aa
        2 2  b  2  2  ba
        3 3  c  3  3  ca
        4 3  d  4  3  da
        5 4  e  5  4  ea
        1 4  f  1  4  fa
        2 5  g  2  5  ga
        3 5  h  3  5  ha
        4 6  i  4  6  ia
        5 6  j  5  6  ja
    >>> result = merge_on_columns_without_breaking_index(df1, df2, left_on=["A", "B"], right_on=["a", "b"], how="inner",
    ...                                                  reindex_by_left=False, reindex_by_right=True)
    >>> result
    ... # doctest: +NORMALIZE_WHITESPACE
             A  B  C   c
        a b
        1 2  1  2  a  aa
        2 2  2  2  b  ba
        3 3  3  3  c  ca
        4 3  4  3  d  da
        5 4  5  4  e  ea
        1 4  1  4  f  fa
        2 5  2  5  g  ga
        3 5  3  5  h  ha
        4 6  4  6  i  ia
        5 6  5  6  j  ja
    """

    if (on is None) and ((left_on is None) or (right_on is None)):
        raise ValueError("onを指定するか、left_onとright_onの両方を指定する必要があります。")
    if on is not None:
        left_on = on
        right_on = on

    if isinstance(left_df.index, MultiIndex):
        left_name = left_df.index.names
    else:
        left_name = left_df.index.name

    if isinstance(right_df.index, MultiIndex):
        right_name = right_df.index.names
    else:
        right_name = right_df.index.name

    assert reindex_by_left ^ reindex_by_right
    if keep_left_index and keep_right_index:
        assert left_name != right_name

    reindex_by = None

    if reindex_by_left:
        assert left_name is not None
        reindex_by = left_name
    elif reindex_by_right:
        assert right_name is not None
        reindex_by = right_name

    if keep_left_index:
        assert left_name is not None
        left_df = left_df.reset_index()

    if keep_right_index:
        assert right_name is not None
        right_df = right_df.reset_index()

    merge_result = left_df.merge(right_df, left_on=left_on, right_on=right_on, how=how)
    if reindex_by_left or reindex_by_right:
        merge_result = merge_result.set_index(reindex_by)
    return merge_result
