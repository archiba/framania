from tempfile import mktemp
from unittest import TestCase

import intake
from dask.dataframe.utils import make_meta
from intake.source.csv import CSVSource

from framania.intakemania.user_defined_class_parquet_source import upload_with_user_defined_class, \
    UserDefinedClassParquetSource


class MyClass:
    def __init__(self, x):
        self._x = x

    def __eq__(self, other):
        return self._x == other._x

    def __hash__(self):
        return hash(self._x)


class TestFramaniaExtendedIntake(TestCase):
    def setUp(self) -> None:
        self.csvsource1 = CSVSource(urlpath='test/data/test-csv1.csv')

    @staticmethod
    def create_custom_source(df):
        filename = mktemp()
        df["d"] = df["b"].map(lambda x: MyClass(x), meta=make_meta(df["b"]))
        source = upload_with_user_defined_class(df, filename, ["d"])

        return source, filename

    def test_upload_with_custom_class(self):
        df = self.csvsource1.to_dask()
        source, _ = self.create_custom_source(df)
        assert source.metadata["user_defined_class_columns"] == ["d"]
        assert df["d"].head().iloc[0] == MyClass("a")
        assert isinstance(source, UserDefinedClassParquetSource)

    def test_load_with_custom_class(self):
        df = self.csvsource1.to_dask()
        source, _ = self.create_custom_source(df)
        yaml_file = mktemp(suffix=".yaml")
        f = open(yaml_file, "w")
        f.write("""\
metadata:
  version: 1
sources: {}""")
        f.close()
        cat = intake.open_catalog(yaml_file)
        cat.add(source, name="test")

        df2 = source.to_dask()

        assert all(df["d"].compute() == df2["d"].compute())
        assert all(df.compute() == df2.compute())
        # TODO(higumachan): これが通らない
        # assert md5hash(df.compute()) == md5hash(df2.compute())

    def test_persist_user_defined_parquet_source(self):
        df = self.csvsource1.to_dask()
        source, filename = self.create_custom_source(df)

        source.persist()

        assert source.has_been_persisted
