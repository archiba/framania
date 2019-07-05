from collections import OrderedDict
from typing import Any, List, Tuple, Callable

import pandas
from dask.dataframe import DataFrame


def make_meta(index_name_type: Tuple[Any, Any], column_name_types: List[Tuple[Any, Any]]) -> pandas.DataFrame:
    """
    API to make dask _meta from index schema info and column schema info.

    Args:
            index_name_type (Tuple[Any, Any): Tuple like (index_name, index_type). index_name can be None.
            column_name_types (List[Tuple[Any, Any]]): List of Tuple like (column name, column type). The order of it will be kept in meta.
        Returns:
            result meta as an empty pandas DataFrame
        Examples:
            >>> result = make_meta(('idx', 'object'),
            ...                    [('int_value', 'int'), ('str_value', 'object'), ('float_value', 'float')])
            >>> print(result.columns)
            Index(['int_value', 'str_value', 'float_value'], dtype='object')
            >>> print(result.dtypes)
            int_value        int64
            str_value       object
            float_value    float64
            dtype: object
            >>> print(result.index)
            Index([], dtype='object', name='idx')
    """
    index_name, index_type = index_name_type
    idx = pandas.Index(data=[], name=index_name, dtype=index_type)
    columns = OrderedDict([(c_name, pandas.Series(data=[], index=idx, dtype=c_type, name=c_name))
                           for c_name, c_type in column_name_types])
    return pandas.DataFrame(columns, index=idx)


def map_partitions_as_meta(dd: DataFrame, func: Callable[..., pandas.DataFrame],
                           meta: pandas.DataFrame, **kwargs):
    """
    API to do map_partitions, and reformat result using meta. It may avoid error caused by map_partitions result and meta don't match.

    Args:
            dd (DataFrame): dask dataframe to do map_partitions.
            func (Callable[[[pandas.DataFrame, ...]], pandas.DataFrame]): function for map_partitions.
            meta (pandas.DataFrame): expected schema of map_partitions result.
            kwargs: additional arguments for func.
        Returns:
            result dask dataframe
        Examples:
            >>> import dask.dataframe
            >>> import pandas
            >>> pd = pandas.DataFrame({'a1': ['1,2,3', '2,3,4', '3,4'], 'a2': ['a,b,c', 'b,c,d', 'c,d'],
            ...                        'b': [1, 2, 3], 'idx': [0, 1, 2]}).set_index('idx')
            >>> print(pd)
            ... # doctest: +NORMALIZE_WHITESPACE
                    a1     a2  b
            idx
            0    1,2,3  a,b,c  1
            1    2,3,4  b,c,d  2
            2      3,4    c,d  3
            >>> transformer = lambda pd: pd[['b', 'a1', 'a2']]
            >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
            >>> result1 = dd.map_partitions(transformer)
            >>> print(result1.compute())
            ... # doctest: +NORMALIZE_WHITESPACE
                 b     a1     a2
            idx
            0    1  1,2,3  a,b,c
            1    2  2,3,4  b,c,d
            2    3    3,4    c,d
            >>> meta = make_meta(('idx', 'int'), [('a1', 'object'), ('a2', 'object'), ('b', 'int')])
            >>> result2 = map_partitions_as_meta(dd, transformer, meta)
            >>> print(result2.compute())
            ... # doctest: +NORMALIZE_WHITESPACE
                    a1     a2  b
            idx
            0    1,2,3  a,b,c  1
            1    2,3,4  b,c,d  2
            2      3,4    c,d  3
            >>> transformer2 = lambda pd, v: pd[['b']] * v
            >>> result3 = map_partitions_as_meta(dd, transformer2, make_meta(('idx', 'int'), [('b', 'int')]),
            ...                                  v=100)
            >>> print(result3.compute())
            ... # doctest: +NORMALIZE_WHITESPACE
                   b
            idx
            0    100
            1    200
            2    300
    """

    def apply_meta(pd: pandas.DataFrame):
        result = func(pd, **kwargs)
        meta_column_names = [c for c in meta.columns]
        return result[meta_column_names]

    return dd.map_partitions(apply_meta)
