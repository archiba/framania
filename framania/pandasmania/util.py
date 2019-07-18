from typing import Tuple

import numpy


def parse_dtype(dtype: numpy.dtype) -> Tuple[str, int]:
    """
    API to parse dtype used for pandas/dask.

        Args:
            dtype(numpy.dtype): dtype to parse
        Returns:
            dtype kind and dtype alignment
        Examples:
            >>> import pandas
            >>> dtypes = ['int8', 'int16', 'int32', 'int64',
            ...           'uint8', 'uint16', 'uint32', 'uint64',
            ...           'float16', 'float32', 'float64',
            ...           'complex64', 'complex128', 'complex256',
            ...           'str', 'object', 'bool', 'bytes']
            >>> df = pandas.DataFrame({t: pandas.Series(dtype=t) for t in dtypes})
            >>> df['timestamp'] = pandas.to_datetime([])
            >>> df['timedelta'] = pandas.to_timedelta([])
            >>> print(df)
            Empty DataFrame
            Columns: [int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, complex64, complex128, complex256, str, object, bool, bytes, timestamp, timedelta]
            Index: []
            >>> print(df.dtypes)
            int8                     int8
            int16                   int16
            int32                   int32
            int64                   int64
            uint8                   uint8
            uint16                 uint16
            uint32                 uint32
            uint64                 uint64
            float16               float16
            float32               float32
            float64               float64
            complex64           complex64
            complex128         complex128
            complex256         complex256
            str                    object
            object                 object
            bool                     bool
            bytes                  object
            timestamp      datetime64[ns]
            timedelta     timedelta64[ns]
            dtype: object
            >>> for dtype in df.dtypes:
            ...     print(parse_dtype(dtype))
            ('i', 1)
            ('i', 2)
            ('i', 4)
            ('i', 8)
            ('u', 1)
            ('u', 2)
            ('u', 4)
            ('u', 8)
            ('f', 2)
            ('f', 4)
            ('f', 8)
            ('c', 4)
            ('c', 8)
            ('c', 16)
            ('O', 8)
            ('O', 8)
            ('b', 1)
            ('O', 8)
            ('M', 8)
            ('m', 8)

    """
    return (dtype.kind, dtype.alignment)
