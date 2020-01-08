import hashlib
from typing import Union, Dict, Any, Optional

import pandas
from dask.dataframe import DataFrame, Series

from framania.pandasmania.util import md5hash as md5hash_pandas


def dataframe_from_series_of_pandas(series_of_pandas_dataframes: Series,
                                    schema: Optional[Union[pandas.DataFrame, DataFrame, Dict[Any, Any]]] = None):
    """
    pandas.DataFrameが各行に保持されているdask.dataframe.Seriesをdask.dataframe.Dataframeに変換するAPI。
    schemaを指定すればそれを結果のスキーマとして使用し、Noneなら第１パーティションの第１行に保持されているdataframeを使用する。
    series内のdataframeのindexは無視され、元のseriesのindexで上書きされる。

    Args:
        series_of_pandas_dataframes: dask dataframe from series of pandas dataframes.
        schema: schema of resutl dataframe
    Return: dask dataframe
    Examples:
        >>> import dask.dataframe
        >>> import pandas
        >>> df = pandas.DataFrame({'a': range(100), 'b': range(100, 200)})
        >>> print(df)
        ... # doctest: +NORMALIZE_WHITESPACE
             a    b
        0    0  100
        1    1  101
        2    2  102
        3    3  103
        4    4  104
        ..  ..  ...
        95  95  195
        96  96  196
        97  97  197
        98  98  198
        99  99  199
        [100 rows x 2 columns]
        >>> ddf = dask.dataframe.from_pandas(df, npartitions=4)
        >>> def build_df_from_row(row):
        ...     pdf = pandas.DataFrame({'values': [row.a, row.a + 1, row.a + 2], 'support': [row.b, row.b, row.b]})
        ...     return pdf
        >>> df_ds = ddf.apply(build_df_from_row, axis=1)
        >>> print(df_ds)
        ... # doctest: +NORMALIZE_WHITESPACE
        Dask Series Structure:
        npartitions=4
        0     object
        25       ...
        50       ...
        75       ...
        99       ...
        dtype: object
        Dask Name: apply, 8 tasks
        >>> result = dataframe_from_series_of_pandas(df_ds)
        >>> print(result)
        ... # doctest: +NORMALIZE_WHITESPACE
        Dask DataFrame Structure:
                      values support
        npartitions=4
        0              int64   int64
        25               ...     ...
        50               ...     ...
        75               ...     ...
        99               ...     ...
        Dask Name: create_pandas_dataframe_in_partition, 12 tasks
        >>> print(result.compute())
        ... # doctest: +NORMALIZE_WHITESPACE
            values  support
        0        0      100
        0        1      100
        0        2      100
        1        1      101
        1        2      101
        ..     ...      ...
        98      99      198
        98     100      198
        99      99      199
        99     100      199
        99     101      199
        [300 rows x 2 columns]
    """
    if schema is None:
        schema = series_of_pandas_dataframes.head(1).iloc[0]

    def create_pandas_dataframe_in_partition(series_chunk: pandas.Series):
        for i, v in series_chunk.iteritems():
            v.set_axis([i] * len(v.index), axis=0, inplace=True)
        df = pandas.concat(list(series_chunk), axis=0)
        return df

    ddf = series_of_pandas_dataframes.map_partitions(create_pandas_dataframe_in_partition, meta=schema)
    return ddf


def dataframe_from_series_of_record_dict(series_of_record_dicts: Series,
                                         schema: Union[pandas.DataFrame, DataFrame, Dict[Any, Any]]):
    """
    Dictが各行に保持されているdask.dataframe.Seriesをdask.dataframe.Dataframeに変換するAPI。
    schemaを指定すればそれを結果のスキーマとして使用し、Noneなら第１パーティションの第１行に保持されているdataframeを使用する。

    Args:
        series_of_record_dicts: dask dataframe from series of pandas dataframes.
        schema: schema of resutl dataframe
    Return: dask dataframe
    Examples:
        >>> import dask.dataframe
        >>> import pandas
        >>> df = pandas.DataFrame({'a': range(100), 'b': range(100, 200)})
        >>> print(df)
        ... # doctest: +NORMALIZE_WHITESPACE
             a    b
        0    0  100
        1    1  101
        2    2  102
        3    3  103
        4    4  104
        ..  ..  ...
        95  95  195
        96  96  196
        97  97  197
        98  98  198
        99  99  199
        [100 rows x 2 columns]
        >>> ddf = dask.dataframe.from_pandas(df, npartitions=4)
        >>> def build_df_from_row(row):
        ...     return [{'values': row.a, 'support': row.b},
        ...             {'values': row.a + 1, 'support': row.b},
        ...             {'values': row.a + 2, 'support': row.b}]
        >>> df_ds = ddf.apply(build_df_from_row, axis=1, meta='object')
        >>> print(df_ds)
        ... # doctest: +NORMALIZE_WHITESPACE
        Dask Series Structure:
        npartitions=4
        0     object
        25       ...
        50       ...
        75       ...
        99       ...
        dtype: object
        Dask Name: apply, 8 tasks
        >>> result = dataframe_from_series_of_record_dict(df_ds, {'values': 'int', 'support': 'int'})
        >>> print(result)
        ... # doctest: +NORMALIZE_WHITESPACE
        Dask DataFrame Structure:
                      values support
        npartitions=4
        0              int64   int64
        25               ...     ...
        50               ...     ...
        75               ...     ...
        99               ...     ...
        Dask Name: create_pandas_dataframe_in_partition, 12 tasks
        >>> print(result.compute())
        ... # doctest: +NORMALIZE_WHITESPACE
            values  support
        0        0      100
        0        1      100
        0        2      100
        1        1      101
        1        2      101
        ..     ...      ...
        98      99      198
        98     100      198
        99      99      199
        99     100      199
        99     101      199
        [300 rows x 2 columns]
    """
    def create_pandas_dataframe_in_partition(series_chunk: pandas.Series):
        records = []
        index = []
        for i, v in series_chunk.iteritems():
            records.extend(v)
            index.extend([i] * len(v))
        df = pandas.DataFrame.from_records(records, index=index)
        return df

    ddf = series_of_record_dicts.map_partitions(create_pandas_dataframe_in_partition, meta=schema)
    return ddf


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
