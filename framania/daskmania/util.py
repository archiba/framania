import hashlib

from dask.dataframe import DataFrame

from framania.pandasmania.util import md5hash as md5hash_pandas


def md5hash(dd: DataFrame) -> hashlib.md5:
    """
    API to make md5hash from pandas dataframe.

    Args:
        pd: pandas dataframe.
    Examples:
        >>> import dask.dataframe
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
        >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
        >>> md5hash(dd)
        'e650778d6cdca953fad30cccedd4c5f1'
    """
    results = dd.map_partitions(md5hash_pandas).compute()
    m = hashlib.md5()
    for v in results:
        m.update(v.encode('utf-8'))
    return m.hexdigest()

