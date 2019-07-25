import os
import shutil
from pathlib import Path
from unittest import TestCase

import dask.dataframe
from intake.source.csv import CSVSource

from framania.intakemania.extention import FramaniaExtendedIntakeCatalog, analysis


class TestFramaniaExtendedIntake(TestCase):
    def setUp(self) -> None:
        self.csvsource1 = CSVSource(urlpath='test/data/test-csv1.csv')
        self.csvsource2 = CSVSource(urlpath='test/data/test-csv2.csv')
        self.cpath = Path('test/temp/test-catalog.yaml')
        self.catalog = FramaniaExtendedIntakeCatalog(self.cpath)
        self.data_dir = Path('test/temp/data-dir')

    def tearDown(self) -> None:
        os.remove(str(self.cpath))
        shutil.rmtree(str(self.data_dir))

    def test_initialize(self):

        @analysis('raw_csv1', '1.0', catalog=self.catalog, sources=[], data_dir=self.data_dir)
        def load_raw_csv1(csv_source1):
            return csv_source1

        load_raw_csv1(self.csvsource1.to_dask())

        source = self.catalog['raw_csv1_1.0']
        intake_source = source.intake_source
        assert intake_source.metadata['extension'] == 'framania'
        assert intake_source.metadata['version'] == source.version == '1.0'
        assert intake_source.metadata['version_name'] == 'raw_csv1_1.0'
        assert intake_source.metadata['name'] == source.name == 'raw_csv1'
        assert intake_source.metadata['upstream'] == []
        assert intake_source.metadata['md5hash'] == source.md5hash == 'd5a5c2bb9d4281f1b0e55337355d288a'

        assert source.upstream_sources == []

    def test_compare(self):
        @analysis('raw_csv1', '1.0', catalog=self.catalog, sources=[], data_dir=self.data_dir)
        def load_raw_csv1(csv_source1):
            return csv_source1

        raw_csv1_source, _ = load_raw_csv1(self.csvsource1.to_dask())

        @analysis('raw_csv2', '1.0', catalog=self.catalog, sources=[], data_dir=self.data_dir)
        def load_raw_csv2(csv_source2):
            return csv_source2

        raw_csv2_source, _ = load_raw_csv2(self.csvsource2.to_dask())

        assert raw_csv1_source.md5hash != raw_csv2_source.md5hash

        source = self.catalog['raw_csv2_1.0']
        intake_source = source.intake_source
        assert intake_source.metadata['extension'] == 'framania'
        assert intake_source.metadata['version'] == source.version == '1.0'
        assert intake_source.metadata['version_name'] == 'raw_csv2_1.0'
        assert intake_source.metadata['name'] == source.name == 'raw_csv2'
        assert intake_source.metadata['upstream'] == []
        assert intake_source.metadata['md5hash'] == source.md5hash == 'd71557c92f2ed1846652c7ce769a4b7e'

        assert source.upstream_sources == []

    def test_dag(self):
        @analysis('raw_csv1', '1.0', catalog=self.catalog, sources=[], data_dir=self.data_dir)
        def load_raw_csv1(csv_source1):
            return csv_source1

        raw_csv1_source, _ = load_raw_csv1(self.csvsource1.to_dask())

        @analysis('raw_csv2', '1.0', catalog=self.catalog, sources=[], data_dir=self.data_dir)
        def load_raw_csv2(csv_source2):
            return csv_source2

        raw_csv2_source, _ = load_raw_csv2(self.csvsource2.to_dask())

        @analysis('transform_csv1', '1.0', catalog=self.catalog, sources=[raw_csv1_source], data_dir=self.data_dir)
        def transform_csv1(raw_csv1):
            raw_csv1['a'] *= 10
            return raw_csv1

        @analysis('transform_csv1_2', '1.0', catalog=self.catalog, sources=[raw_csv1_source, raw_csv2_source],
                  data_dir=self.data_dir)
        def transform_csv1_2(raw_csv1, raw_csv2):
            raw_csv1_rename = raw_csv1.rename(columns={'a': 'a1'})
            raw_csv2_rename = raw_csv2.rename(columns={'a': 'a2'})
            return dask.dataframe.concat([raw_csv1_rename[['a1']], raw_csv2_rename[['a2']]], axis=1,
                                         interleave_partitions=True)
        transform_csv1()

        source = self.catalog['transform_csv1_1.0']
        intake_source = source.intake_source
        assert intake_source.metadata['extension'] == 'framania'
        assert intake_source.metadata['version'] == source.version == '1.0'
        assert intake_source.metadata['version_name'] == 'transform_csv1_1.0'
        assert intake_source.metadata['name'] == source.name == 'transform_csv1'
        assert intake_source.metadata['upstream'] == [{'version_name': 'raw_csv1_1.0',
                                                       'md5hash': 'd5a5c2bb9d4281f1b0e55337355d288a'}]
        assert intake_source.metadata['md5hash'] == source.md5hash == 'd71557c92f2ed1846652c7ce769a4b7e'

        assert source.upstream_sources == [self.catalog['raw_csv1_1.0']]

        transform_csv1_2()

        source = self.catalog['transform_csv1_2_1.0']
        intake_source = source.intake_source
        assert intake_source.metadata['extension'] == 'framania'
        assert intake_source.metadata['version'] == source.version == '1.0'
        assert intake_source.metadata['version_name'] == 'transform_csv1_2_1.0'
        assert intake_source.metadata['name'] == source.name == 'transform_csv1_2'
        assert intake_source.metadata['upstream'] == [{'version_name': 'raw_csv1_1.0',
                                                       'md5hash': 'd5a5c2bb9d4281f1b0e55337355d288a'},
                                                      {'version_name': 'raw_csv2_1.0',
                                                       'md5hash': 'd71557c92f2ed1846652c7ce769a4b7e'}]
        assert intake_source.metadata['md5hash'] == source.md5hash == 'e875d34af26e1db441c33b86ed158e51'

        assert source.upstream_sources == [self.catalog['raw_csv1_1.0'], self.catalog['raw_csv2_1.0']]

        assert self.catalog.validate()[0]

    def test_validate(self):
        @analysis('raw_csv1', '1.0', catalog=self.catalog, sources=[], data_dir=self.data_dir)
        def load_raw_csv1(csv_source1):
            return csv_source1

        raw_csv1_source, _ = load_raw_csv1(self.csvsource1.to_dask())

        @analysis('transform_csv1', '1.0', catalog=self.catalog, sources=[raw_csv1_source], data_dir=self.data_dir)
        def transform_csv1(raw_csv1):
            raw_csv1['a'] *= 10
            return raw_csv1

        transform_csv1()

        source = self.catalog['transform_csv1_1.0']
        intake_source = source.intake_source
        assert intake_source.metadata['extension'] == 'framania'
        assert intake_source.metadata['version'] == source.version == '1.0'
        assert intake_source.metadata['version_name'] == 'transform_csv1_1.0'
        assert intake_source.metadata['name'] == source.name == 'transform_csv1'
        assert intake_source.metadata['upstream'] == [{'version_name': 'raw_csv1_1.0',
                                                       'md5hash': 'd5a5c2bb9d4281f1b0e55337355d288a'}]
        assert intake_source.metadata['md5hash'] == source.md5hash == 'd71557c92f2ed1846652c7ce769a4b7e'

        assert source.upstream_sources == [self.catalog['raw_csv1_1.0']]

        raw_csv1_source, _ = load_raw_csv1(self.csvsource2.to_dask())
        print(self.catalog.validate())
        assert not self.catalog.validate()[0]
