from pathlib import Path
from typing import Union
from urllib.parse import urlparse
from uuid import uuid4

import dask
import pandas
import yaml
from dask.dataframe import DataFrame
from intake import DataSource, open_catalog
from intake.catalog import Catalog
from intake.catalog.local import YAMLFileCatalog
from intake_parquet import ParquetSource


class S3URL(str):
    """
    pathlib.Path like s3 url wrapper class.
    """

    def __new__(cls, content: str):
        if not content.startswith('s3://'):
            content = f's3://{content}'
        return str.__new__(cls, content)

    def __truediv__(self, other):
        if not self.endswith('/'):
            return S3URL(f'{self}/{other}')
        else:
            return S3URL(f'{self}{other}')

    def bucket(self):
        parsed = urlparse(self)
        return parsed.netloc

    def key(self):
        parsed = urlparse(self)
        return parsed.path.lstrip('/')

    def mkdir(self, **kwargs):
        pass


def local_or_s3_path(v: Union[str, Path, S3URL]) -> Union[Path, S3URL]:
    """
    Convert input value into local path(pathlib.Path) or S3URL.
    When yes, then return input value.
    When no and value starts with s3://, then return s3 remote url.
    Other case return local path

    Args:
        v: str, local path, or s3 remote url.
    Returns:
        local path or s3 remote url
    Examples:
        >>> assert isinstance(local_or_s3_path(Path('abc')), Path)
        >>> assert isinstance(local_or_s3_path(S3URL('abc')), S3URL)
        >>> assert isinstance(local_or_s3_path('abc'), Path)
        >>> assert isinstance(local_or_s3_path('s3://abc'), S3URL)
    """
    if isinstance(v, Path):
        return v
    if isinstance(v, S3URL):
        return v
    if isinstance(v, str) and v.startswith('s3://'):
        return S3URL(v)
    else:
        return Path(v)


def initialize_catalog(catalog_file: Union[Path, S3URL, str]):
    catalog_file = local_or_s3_path(catalog_file)
    try:
        return open_catalog(str(catalog_file))
    except FileNotFoundError:
        catalog: Catalog = open_catalog()
        catalog.save(str(catalog_file))
        return catalog


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
        >>> wd = os.getcwd()
        >>> source1 = CSVSource('test/temp/test1.csv')
        >>> source1.name = 'csv-test1'
        >>> # create new catalog with source
        >>> cfile = 'test/temp/test-catalog.yaml'
        >>> add_source_to_catalog(source1, cfile)
        >>> print(yaml.safe_load(Path(cfile).open().read().replace(wd, '')))
        ... # doctest: +NORMALIZE_WHITESPACE
        {'metadata': {}, 'sources': {'csv-test1': {'args': {'urlpath': 'test/temp/test1.csv'}, 'description': '', 'driver': 'intake.source.csv.CSVSource', 'metadata': {}}}}
        >>> source2 = CSVSource('test/temp/test2.csv')
        >>> source2.name = 'csv-test2'
        >>> add_source_to_catalog(source2, cfile)
        >>> print(yaml.safe_load(Path(cfile).open().read().replace(wd, '')))
        ... # doctest: +NORMALIZE_WHITESPACE
        {'metadata': {}, 'sources': {'csv-test1': {'args': {'urlpath': 'test/temp/test1.csv'}, 'container': 'dataframe', 'description': '', 'direct_access': 'forbid', 'driver': ['csv'], 'metadata': {}, 'name': 'csv-test1', 'plugin': ['csv'], 'user_parameters': []}, 'csv-test2': {'args': {'urlpath': 'test/temp/test2.csv'}, 'description': '', 'driver': 'intake.source.csv.CSVSource', 'metadata': {}}}}
        >>> os.remove(cfile)
    """
    catalog_file = local_or_s3_path(catalog_file)
    try:
        catalog: YAMLFileCatalog = YAMLFileCatalog(path=str(catalog_file))
    except FileNotFoundError:
        _catalog = open_catalog()
        _catalog.save(url=str(catalog_file))
        catalog: YAMLFileCatalog = YAMLFileCatalog(path=str(catalog_file))
    catalog.add(source, name=source.name)


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
        100    1  2  5     a
        200    2  3  6     b
        300    3  4  7     c
        400    4  5  8     d
        >>> print(len(job) == 1)
        True
        >>> print(job[0] is None)
        True
        >>> print(yaml.safe_load(Path(cfile).open().read()))
        ... # doctest: +NORMALIZE_WHITESPACE
        {'metadata': {}, 'sources': {'test-dd1': {'args': {'urlpath': 'test/temp/data-dir/test-dd1'}, 'description': '', 'driver': 'intake_parquet.source.ParquetSource', 'metadata': {}}}}
        >>> # DUMP WITHOUT COMPUTATION
        >>> psource2, job = dump_dask_to_intake(dd, 'test-dd2', ddir, cfile, compute=False)
        >>> print(job is None)
        False
        >>> # do computation lazily
        >>> job.compute()
        >>> print(psource2.read())
        ... # doctest: +NORMALIZE_WHITESPACE
               a  b  c label
        100    1  2  5     a
        200    2  3  6     b
        300    3  4  7     c
        400    4  5  8     d
        >>> os.remove(cfile)
        >>> shutil.rmtree(ddir)
    """
    data_dir = local_or_s3_path(data_dir)

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
        100    1  2  5     a
        200    2  3  6     b
        300    3  4  7     c
        400    4  5  8     d
        >>> print(len(job) == 1)
        True
        >>> print(job[0] is None)
        True
        >>> print(yaml.safe_load(Path(cfile).open().read()))
        ... # doctest: +NORMALIZE_WHITESPACE
         {'metadata': {}, 'sources': {'test-pd1': {'args': {'urlpath': 'test/temp/data-dir/test-pd1'}, 'description': '', 'driver': 'intake_parquet.source.ParquetSource', 'metadata': {}}}}
        >>> # DUMP WITHOUT COMPUTATION
        >>> psource2, job = dump_pandas_to_intake(pd, 'test-pd2', ddir, cfile, compute=False)
        >>> print(job is None)
        False
        >>> # do computation lazily
        >>> job.compute()
        >>> print(psource2.read())
        ... # doctest: +NORMALIZE_WHITESPACE
               a  b  c label
        100    1  2  5     a
        200    2  3  6     b
        300    3  4  7     c
        400    4  5  8     d
        >>> os.remove(cfile)
        >>> shutil.rmtree(ddir)
    """
    dd = dask.dataframe.from_pandas(pd, npartitions=npartitions)
    return dump_dask_to_intake(dd, data_name, data_dir, catalog_file, **kwargs)


def persist_local(dd: DataFrame, persist_dir: Union[str, Path], **kwargs):
    """
    API to persist dask dataframe as perquet files on disk.

    Args:
        dd: dask dataframe to persist.
        persist_dir: directory where dask dataframe will be stored.
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
        ... # doctest: +NORMALIZE_WHITESPACE
             a  b  c label
        100  1  2  5     a
        200  2  3  6     b
        300  3  4  7     c
        400  4  5  8     d
        >>> dd = dask.dataframe.from_pandas(pd, npartitions=2)
        >>> ddir = 'test/temp/data-dir/persist'
        >>> dd2 = persist_local(dd, ddir)
        >>> persist_id = dd2.persist_id
        >>> print(dd2.compute())
        ... # doctest: +NORMALIZE_WHITESPACE
               a  b  c label
        100    1  2  5     a
        200    2  3  6     b
        300    3  4  7     c
        400    4  5  8     d
        >>> assert (Path(ddir) / persist_id).exists()
        >>> shutil.rmtree(ddir)
    """
    data_id = str(uuid4())
    persist_dir = local_or_s3_path(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    persist_catalog_file = persist_dir / 'persist_catalog.yaml'
    psource, _ = dump_dask_to_intake(dd, data_id, persist_dir, persist_catalog_file, **kwargs)

    new_dd = psource.to_dask()
    new_dd.persist_id = data_id

    return new_dd
