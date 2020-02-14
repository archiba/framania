from hashlib import sha1
from typing import List

import numpy
import pandas
from pandas import Series


def str_startswith_many(ps: Series, startswiths: List[str]):
    """
        API to check string column values start with some of specified string sequences.

        Args:
            ps (pandas.Series): target pandas series
            startswiths (List[str]): string sequences to check string column startswith
        Returns:
            bool pandas series
        Examples:
            >>> import pandas
            >>> pd = pandas.DataFrame({'a': ['abc', 'bcd', 'cde', 'def']})
            >>> print(pd)
            ... # doctest: +NORMALIZE_WHITESPACE
                 a
            0  abc
            1  bcd
            2  cde
            3  def
            >>> result = str_startswith_many(pd.a, ['ab', 'bc'])
            >>> print(result)
            ... # doctest: +NORMALIZE_WHITESPACE
            0     True
            1     True
            2    False
            3    False
            dtype: bool
        """
    result = pandas.Series(False, index=ps.index)
    for swith in startswiths:
        result |= ps.str.startswith(swith)
    return result


def str_endswith_many(ps: Series, endswiths: List[str]):
    """
        API to check string column values end with some of specified string sequences.

        Args:
            ps (pandas.Series): target pandas series
            endswiths (List[str]): string sequences to check string column endswith
        Returns:
            bool pandas series
        Examples:
            >>> import pandas
            >>> pd = pandas.DataFrame({'a': ['abc', 'bcd', 'cde', 'def']})
            >>> print(pd)
            ... # doctest: +NORMALIZE_WHITESPACE
                 a
            0  abc
            1  bcd
            2  cde
            3  def
            >>> result = str_endswith_many(pd.a, ['de', 'ef'])
            >>> print(result)
            ... # doctest: +NORMALIZE_WHITESPACE
            0    False
            1    False
            2     True
            3     True
            dtype: bool
        """
    result = pandas.Series(False, index=ps.index)
    for ewith in endswiths:
        result |= ps.str.endswith(ewith)
    return result


def str_contains_many(ps: Series, contains: List[str]):
    """
        API to check string column values contain some of specified string sequences.

        Args:
            ps (pandas.Series): target pandas series
            contains (List[str]): string sequences to check string column contains
        Returns:
            bool pandas series
        Examples:
            >>> import pandas
            >>> pd = pandas.DataFrame({'a': ['abc', 'bcd', 'cde', 'def']})
            >>> print(pd)
            ... # doctest: +NORMALIZE_WHITESPACE
                 a
            0  abc
            1  bcd
            2  cde
            3  def
            >>> result = str_contains_many(pd.a, ['abc', 'def'])
            >>> print(result)
            ... # doctest: +NORMALIZE_WHITESPACE
            0     True
            1    False
            2    False
            3     True
            dtype: bool
        """
    result = pandas.Series(False, index=ps.index)
    for c in contains:
        result |= ps.str.contains(c)
    return result


def _hash(s, hash_class=None, dtype=numpy.uint64):
    return dtype(int.from_bytes(hash_class(s.encode("utf8")).digest()[:8], "little", signed=False))


def str_to_uint64(ps: Series):
    """
    API to mapping str to uint64 with hashing function.
    Args:
        ps (pandas.Series): target pandas series
    Returns:
        uint64 pandas series
    Examples:
        >>> import pandas
        >>> series = pandas.Series(["nadeko", "test", "nadeko"], dtype=numpy.object)
        >>> processed = str_to_uint64(series)
        >>> assert processed.dtype == numpy.uint64
        >>> assert processed[0] == processed[2]
        >>> assert processed[0] != processed[1]
    """

    return ps.apply(_hash, hash_class=sha1, dtype=numpy.uint64)
