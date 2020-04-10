import re
from pathlib import Path
from typing import List, Optional, Any, Union, Dict, Tuple

import pandas
from dask.dataframe import DataFrame, from_pandas
from intake import DataSource
from intake.catalog import Catalog
from intake_parquet import ParquetSource

from framania.daskmania.util import md5hash as ddmd5hash
from framania.intakemania.util import add_source_to_catalog, initialize_catalog, local_or_s3_path, S3URL


def parse_version(v: Any):
    if not isinstance(v, str):
        return v

    version_info = tuple(part if not part.isnumeric() else int(part) for part in v.split('.'))
    return version_info


def get_version_name(name: str, version: str):
    return f'{name}_{version}'


def version_and_name(version_name: str):
    if '_' not in version_name:
        raise Exception(f'Invalid version_name {version_name}.')

    split = version_name.rsplit('_', maxsplit=1)
    return split[0], split[1]


def validate_version(version: str):
    version.replace('_', '.')


class FramaniaExtendedIntakeCatalog:
    def __init__(self, intake_catalog_file: Union[Path, S3URL, str]):
        self.intake_catalog = initialize_catalog(intake_catalog_file)
        self.path = intake_catalog_file

    def find_by_version_name(self, version_name: str) -> 'FramaniaExtendedIntakeSource':
        source = self.intake_catalog[version_name]

        source_extension = source.metadata.get('extension', None)
        if source_extension != 'framania':
            raise Exception(f'Data source {version_name} is not framania extended.')

        name, version = version_and_name(version_name)
        assert version == source.metadata['version']
        md5hash = source.metadata['md5hash']

        upstream_sources = [self.find_validated_by_version_name(s['version_name'], s['md5hash']) for s in
                            source.metadata['upstream']]
        return FramaniaExtendedIntakeSource(source, name, version, md5hash, upstream_sources, existing_source=True)

    def find_by_version_and_name(self, name: str, version: str) -> 'FramaniaExtendedIntakeSource':
        return self.find_by_version_name(get_version_name(name, version))

    def find_validated_by_version_name(self, version_name: str, md5hash: str) -> 'FramaniaExtendedIntakeSource':
        found = self.find_by_version_name(version_name)
        assert found.md5hash == md5hash
        return found

    def find_latest_source_by_name(self, name: str) -> 'FramaniaExtendedIntakeSource':
        sources = self.intake_catalog._entries
        source_names = [k for k in sources.keys() if k.startswith(name)]

        versions = {}

        for source_name in source_names:
            source: DataSource = self.intake_catalog[source_name]
            source_version = source.metadata['version']

            if source_name != get_version_name(name, source_version):
                continue

            versions[source_version] = source_name

        latest_version = list(sorted(versions.keys(), key=lambda v: parse_version(v), reverse=True))[0]
        return self.find_by_version_name(versions[latest_version])

    def __getitem__(self, item: Union[str, 'FramaniaExtendedIntakeSource']) -> 'FramaniaExtendedIntakeSource':
        if isinstance(item, FramaniaExtendedIntakeSource):
            item = get_version_name(item.name, item.version)

        try:
            return self.find_by_version_name(item)
        except:
            return self.find_latest_source_by_name(item)

    def append(self, value: 'FramaniaExtendedIntakeSource'):
        add_source_to_catalog(value.intake_source, self.path)
        self.intake_catalog = initialize_catalog(self.path)

    def validate(self) -> Tuple[bool, Dict]:
        result = {}
        result_flag = True

        all_entries = list(self.intake_catalog._entries.keys())

        finished = set()
        l_entries = len(all_entries)
        for i in range(l_entries):
            for j in range(l_entries):
                entry = all_entries[j]

                if entry in finished:
                    continue

                source = self.intake_catalog[entry]
                if source.metadata.get('extension', '') != 'framania':
                    continue

                if any([s['version_name'] not in finished for s in source.metadata['upstream']]):
                    continue

                result[entry] = []

                validate_flag = True
                name, version = version_and_name(entry)
                validate_flag = validate_flag and (source.metadata.get('name', '') == name)
                validate_flag = validate_flag and (source.metadata.get('version', '') == version)
                validate_flag = validate_flag and (source.metadata.get('version_name', '') == entry)

                for s in source.metadata['upstream']:
                    try:
                        self.find_validated_by_version_name(s['version_name'], s['md5hash'])
                        flag = result[s['version_name']][-1][1]
                    except AssertionError:
                        flag = False
                    validate_flag = validate_flag and flag
                    result[entry].append((s['version_name'], flag))

                result[entry].append((entry, validate_flag))
                result_flag = result_flag and validate_flag
                finished.add(entry)
        return result_flag, result

    def dump_dask(self, dd: DataFrame, data_name: str, version: str,
                  data_dir: Union[str, Path], upstream_sources: List['FramaniaExtendedIntakeSource'] = None,
                  **kwargs):
        data_dir = local_or_s3_path(data_dir)

        parquet_dir = data_dir / data_name / version

        parquet_kwargs = {}
        if 'engine' in kwargs:
            parquet_kwargs['engine'] = kwargs['engine']

        to_parquet_result = dd.to_parquet(str(parquet_dir), **kwargs)

        psource = ParquetSource(str(parquet_dir), **parquet_kwargs)
        md5hash = ddmd5hash(psource.to_dask())
        framania_psource = FramaniaExtendedIntakeSource(psource, data_name, version,
                                                        md5hash, upstream_sources)
        self.append(framania_psource)

        return framania_psource, to_parquet_result

    def dump_pandas(self, pd: pandas.DataFrame, data_name: str, version: str,
                    data_dir: Union[str, Path], upstream_sources: List['FramaniaExtendedIntakeSource'] = None,
                    npartitions: int = 1, **kwargs):
        dd = from_pandas(pd, npartitions=npartitions)
        self.dump_dask(dd, data_name, version, data_dir, upstream_sources, **kwargs)


class FramaniaExtendedIntakeSource:
    good_name_chars = '[0-9|a-z|A-Z|\-|_]+'
    good_version_chars = '[0-9|a-z|A-Z|\-|.]+'

    def __init__(self, intake_source: DataSource,
                 name: str, version: str,
                 md5hash: Optional[str] = None,
                 upstream_sources: Optional[List['FramaniaExtendedIntakeSource']] = None,
                 existing_source: bool = False,
                 metadata: Dict[Any, Any] = None):

        if re.match(self.good_name_chars, name) is None:
            raise Exception('Valid data name characters: [0-9, a-z, A-Z, -, _].')
        if re.match(self.good_version_chars, version) is None:
            raise Exception('Valid data version characters: [0-9, a-z, A-Z, -, .].')

        if metadata is None:
            metadata = {}

        self.intake_source = intake_source
        self.name = name
        self.version = version
        self.md5hash = md5hash
        self.upstream_sources = upstream_sources or []

        if not existing_source:
            vname = get_version_name(self.name, self.version)
            self.intake_source.name = vname
            self.intake_source.metadata['version'] = self.version
            self.intake_source.metadata['extension'] = 'framania'
            self.intake_source.metadata['name'] = self.name
            self.intake_source.metadata['version_name'] = vname
            self.intake_source.metadata['md5hash'] = self.md5hash

            upstream_sources = [{'version_name': get_version_name(source.name, source.version),
                                 'md5hash': source.md5hash}
                                for source in self.upstream_sources]

            self.intake_source.metadata['upstream'] = upstream_sources
            self.intake_source.metadata.update(metadata)

    def update_md5hash(self, md5hash):
        self.md5hash = md5hash
        self.intake_source.metadata['md5hash'] = md5hash

    def __eq__(self, other):
        return (self.name == other.name) and \
               (self.version == other.version) and \
               (self.md5hash == other.md5hash) and \
               (self.upstream_sources == other.upstream_sources)


def analysis(name: str, version: str,
             catalog: FramaniaExtendedIntakeCatalog, sources: List[str],
             data_dir: Path):
    """
    Examples:
        >>> import pandas
        >>> import dask.dataframe
        >>> from pathlib import Path
        >>> import os
        >>> import shutil
        >>> ddir = Path('test/temp/data-dir')
        >>> cfile = Path('test/temp/test-catalog.yaml')
        >>> catalog = FramaniaExtendedIntakeCatalog(cfile)
        >>> @analysis('test1', '1.0', catalog, [], ddir)
        ... def test1():
        ...     df = pandas.DataFrame({'a': [1,2,3], 'b': [2,3,4]})
        ...     ddf = dask.dataframe.from_pandas(df, npartitions=1)
        ...     return ddf
        >>> @analysis('test2', '1.0', catalog, ['test1'], ddir)
        ... def test2(test1):
        ...     test1['c'] = test1.a * test1.b
        ...     return test1
        >>> _ = test1()
        >>> _ = test2()
        >>> os.remove(str(cfile))
        >>> shutil.rmtree(str(ddir))
    """

    def decorator_analysis(func):
        def wrapper(*args, **kwargs) -> Tuple[FramaniaExtendedIntakeSource, Any]:
            upstream_sources = [catalog[source] for source in sources]
            input: Dict[str, FramaniaExtendedIntakeSource] = \
                {source.name: source.intake_source.to_dask() for source in upstream_sources}
            result_dask = func(*args, **kwargs, **input)
            result_source, dask_job = catalog.dump_dask(result_dask, name, version, data_dir, upstream_sources)
            return result_source, dask_job

        return wrapper

    return decorator_analysis
