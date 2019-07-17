from unittest import TestCase


class TestIntakeAPI(TestCase):
    def test_import(self):
        import framania.intakemania.api as fiapi
        print(fiapi.add_source_to_catalog)

class TestDaskAPI(TestCase):
    def test_import(self):
        import framania.daskmania.api as fdapi
        print(fdapi.drop_duplicates_by_named_index_and_keys)

class TestPandasAPI(TestCase):
    def test_import(self):
        import framania.pandasmania.api as fpapi
        print(fpapi.drop_duplicates_by_named_index_and_keys)
