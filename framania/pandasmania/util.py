import hashlib

import pandas
from pandas import DataFrame


def md5hash(pd: DataFrame) -> hashlib.md5:
    """
    API to make md5hash from pandas dataframe.

    Args:
        pd: pandas dataframe.
    Examples:
        >>> import pandas
        >>> pd = pandas.DataFrame({'a': [1, 10, 100, 1, 1, 100], 'b': ['a', 'b', 'c', 'd', 'e', 'f'],
        ...                       'idx': [0, 1, 2, 0, 1, 2]}).set_index('idx')
        >>> print(pd)
        ... # doctest: +NORMALIZE_WHITESPACE
               a  b
        idx
        0      1  a
        1     10  b
        2    100  c
        0      1  d
        1      1  e
        2    100  f
        >>> md5hash(pd)
        '80254916224f25f3701c2edc8afa8539'
    """
    hashes = pandas.util.hash_pandas_object(pd)
    m = hashlib.md5()
    for hash in hashes:
        m.update(hash.to_bytes(64, 'big'))
    return m.hexdigest()
