import numpy
from dask.dataframe import Series
from framania.pandasmania.string import str_to_uint64 as pandas_str_to_uint64


def str_to_uint64(ds: Series):
    """
    API to mapping str to uint64 with hashing function.
    Args:
        ds (dask.Series): target pandas series
    Returns:
        uint64 dask series
    Examples:

        >>> from framania.daskmania.string import str_to_uint64
        >>> import dask.dataframe as dd
        >>> import pandas
        >>> series = dd.from_pandas(pandas.Series(["nadeko", "test", "nadeko"], dtype=numpy.object), npartitions=1)
        >>> processed = str_to_uint64(series).compute()
        >>> assert processed.dtype == numpy.uint64
        >>> assert processed[0] == processed[2]
        >>> assert processed[0] != processed[1]
    """

    return ds.map_partitions(pandas_str_to_uint64, meta=(None, numpy.uint64))
