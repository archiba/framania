from pathlib import Path
from typing import Union

import dask
import pandas
import yaml
from dask.dataframe import DataFrame
from intake import DataSource
from intake_parquet import ParquetSource


def add_source_to_catalog(source: DataSource, catalog_file: Union[Path, str]):
    """
    API to add new data source to catalog_file.

    Args:
            source: data source to add.
            catalog_file: file where data source to be added. if file doesn't exist, file will be created.
        Examples:
            >>> import os
            >>> from intake.source.csv import CSVSource
            >>> import yaml
            >>> source1 = CSVSource('test/temp/test1.csv')
            >>> source1.name = 'csv-test1'
            >>> # create new catalog with source
            >>> cfile = 'test/temp/test-catalog.yaml'
            >>> add_source_to_catalog(source1, cfile)
            >>> print(yaml.safe_load(Path(cfile).open().read()))
            ... # doctest: +NORMALIZE_WHITESPACE
            {'sources': {'csv-test1': {'args': {'urlpath': 'test/temp/test1.csv'}, 'description': '', 'driver': 'intake.source.csv.CSVSource', 'metadata': {}}}}
            >>> source2 = CSVSource('test/temp/test2.csv')
            >>> source2.name = 'csv-test2'
            >>> add_source_to_catalog(source2, cfile)
            >>> print(yaml.safe_load(Path(cfile).open().read()))
            ... # doctest: +NORMALIZE_WHITESPACE
            {'sources': {'csv-test1': {'args': {'urlpath': 'test/temp/test1.csv'}, 'description': '', 'driver': 'intake.source.csv.CSVSource', 'metadata': {}}, 'csv-test2': {'args': {'urlpath': 'test/temp/test2.csv'}, 'description': '', 'driver': 'intake.source.csv.CSVSource', 'metadata': {}}}}
            >>> os.remove(cfile)
    """
    if isinstance(catalog_file, str):
        catalog_file = Path(catalog_file)
    try:
        catalog_contents = yaml.load(catalog_file.open().read(), Loader=yaml.SafeLoader)
    except FileNotFoundError:
        catalog_contents = {'sources': {}}
    if catalog_contents is None or \
            not isinstance(catalog_contents['sources'], dict) or \
            'sources' not in catalog_contents:
        catalog_contents = {'sources': {}}
    catalog_contents['sources'][source.name] = source._yaml()['sources'][source.name]

    catalog_file.open('wt').write(yaml.dump(catalog_contents, Dumper=yaml.SafeDumper))


def dump_dask_to_intake(dd: DataFrame, data_name: str, data_dir: Union[str, Path], catalog_file: Union[Path, str],
                        **kwargs):
    """
        API to dump dask dataframe as parquet format and add it to intake catalog

            Args:
                dd: dask dataframe to dump.
                data_name: name used as a name of intake data source.
                data_dir: directory where dask dataframe will be stored.
                catalog_file: file where data source to be added. if file doesn't exist, file will be created.
                kwargs: Any options available for dask.dataframe.to_parquet. see https://docs.dask.org/en/latest/dataframe-api.html#dask.dataframe.to_parquet for detail.
            Returns:
                created parquet data source and dask to_parquet job (if you put compute=False in kwargs.)
            Examples:
                >>> import os
                >>> import shutil
                >>> import yaml
                >>> from intake.source.csv import CSVSource
                >>> import dask.dataframe
                >>> import pandas
                >>> pd = pandas.DataFrame({'a': [1, 2, 3, 4], 'b': [2, 3, 4, 5], 'c': [5, 6, 7, 8],
                ...                        'label': ['a', 'b', 'c', 'd']},
                ...                        index=[100, 200, 300, 400])
                >>> print(pd)
                     a  b  c label
                100  1  2  5     a
                200  2  3  6     b
                300  3  4  7     c
                400  4  5  8     d
                >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
                >>> print(dd)
                ... # doctest: +NORMALIZE_WHITESPACE
                Dask DataFrame Structure:
                                   a      b      c   label
                npartitions=2
                100            int64  int64  int64  object
                300              ...    ...    ...     ...
                400              ...    ...    ...     ...
                Dask Name: from_pandas, 2 tasks
                >>> cfile = 'test/temp/test-catalog.yaml'
                >>> ddir = 'test/temp/data-dir'
                >>> # DUMP WITH COMPUTATION
                >>> psource1, job = dump_dask_to_intake(dd, 'test-dd1', ddir, cfile)
                >>> print(psource1.name)
                test-dd1
                >>> print(psource1.read())
                ... # doctest: +NORMALIZE_WHITESPACE
                       a  b  c label
                index
                100    1  2  5     a
                200    2  3  6     b
                300    3  4  7     c
                400    4  5  8     d
                >>> print(job is None)
                True
                >>> print(yaml.safe_load(Path(cfile).open().read()))
                ... # doctest: +NORMALIZE_WHITESPACE
                {'sources': {'test-dd1': {'args': {'urlpath': 'test/temp/data-dir/test-dd1'}, 'description': '', 'driver': 'intake_parquet.source.ParquetSource', 'metadata': {}}}}
                >>> # DUMP WITHOUT COMPUTATION
                >>> psource2, job = dump_dask_to_intake(dd, 'test-dd2', ddir, cfile, compute=False)
                >>> print(job is None)
                False
                >>> # do computation lazily
                >>> job.compute()
                >>> print(psource2.read())
                ... # doctest: +NORMALIZE_WHITESPACE
                       a  b  c label
                index
                100    1  2  5     a
                200    2  3  6     b
                300    3  4  7     c
                400    4  5  8     d
                >>> os.remove(cfile)
                >>> shutil.rmtree(ddir)
        """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    parquet_dir = data_dir / data_name

    parquet_kwargs = {}
    if 'engine' in kwargs:
        parquet_kwargs['engine'] = kwargs['engine']

    psource = ParquetSource(str(parquet_dir), **parquet_kwargs)
    psource.name = data_name
    add_source_to_catalog(psource, catalog_file)
    to_parquet_result = dd.to_parquet(str(parquet_dir), **kwargs)

    return psource, to_parquet_result


def dump_pandas_to_intake(pd: pandas.DataFrame, data_name: str, data_dir: Union[str, Path],
                          catalog_file: Union[Path, str], npartitions: int = 1, **kwargs):
    """
        API to dump pandas dataframe as parquet format and add it to intake catalog.
        pandas dataframe will be converted to dask dataframe automatically internally.

            Args:
                pd: pandas dataframe to dump.
                data_name: name used as a name of intake data source.
                data_dir: directory where dask dataframe will be stored.
                catalog_file: file where data source to be added. if file doesn't exist, file will be created.
                npartitions: number of partitions input pandas dataframe split into.
                kwargs: Any options available for dask.dataframe.to_parquet. see https://docs.dask.org/en/latest/dataframe-api.html#dask.dataframe.to_parquet for detail.
            Returns:
                created parquet data source and dask to_parquet job (if you put compute=False in kwargs.)
            Examples:
                >>> import os
                >>> import shutil
                >>> import yaml
                >>> from intake.source.csv import CSVSource
                >>> import dask.dataframe
                >>> import pandas
                >>> pd = pandas.DataFrame({'a': [1, 2, 3, 4], 'b': [2, 3, 4, 5], 'c': [5, 6, 7, 8],
                ...                        'label': ['a', 'b', 'c', 'd']},
                ...                        index=[100, 200, 300, 400])
                >>> print(pd)
                     a  b  c label
                100  1  2  5     a
                200  2  3  6     b
                300  3  4  7     c
                400  4  5  8     d
                >>> cfile = 'test/temp/test-catalog.yaml'
                >>> ddir = 'test/temp/data-dir'
                >>> # DUMP WITH COMPUTATION
                >>> psource1, job = dump_pandas_to_intake(pd, 'test-pd1', ddir, cfile)
                >>> print(psource1.name)
                test-pd1
                >>> print(psource1.read())
                ... # doctest: +NORMALIZE_WHITESPACE
                       a  b  c label
                index
                100    1  2  5     a
                200    2  3  6     b
                300    3  4  7     c
                400    4  5  8     d
                >>> print(job is None)
                True
                >>> print(yaml.safe_load(Path(cfile).open().read()))
                ... # doctest: +NORMALIZE_WHITESPACE
                {'sources': {'test-pd1': {'args': {'urlpath': 'test/temp/data-dir/test-pd1'}, 'description': '', 'driver': 'intake_parquet.source.ParquetSource', 'metadata': {}}}}
                >>> # DUMP WITHOUT COMPUTATION
                >>> psource2, job = dump_pandas_to_intake(pd, 'test-pd2', ddir, cfile, compute=False)
                >>> print(job is None)
                False
                >>> # do computation lazily
                >>> job.compute()
                >>> print(psource2.read())
                ... # doctest: +NORMALIZE_WHITESPACE
                       a  b  c label
                index
                100    1  2  5     a
                200    2  3  6     b
                300    3  4  7     c
                400    4  5  8     d
                >>> os.remove(cfile)
                >>> shutil.rmtree(ddir)
        """
    dd = dask.dataframe.from_pandas(pd, npartitions=npartitions)
    return dump_dask_to_intake(dd, data_name, data_dir, catalog_file, **kwargs)
