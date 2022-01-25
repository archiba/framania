from unittest import TestCase
from pandas import DataFrame
import framania.pandasmania.api as fpapi


class TestIntakeAPI(TestCase):
    def test_illoc(self):
        df = DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [3, 4, 5], 'i1': [0, 0, 1], 'i2': [0, 1, 0]})
        df.set_index(['i1', 'i2'], inplace=True)
        assert len(fpapi.illoc(df, 1, 0)) == 2
        assert len(fpapi.illoc(df, 'i2', 0)) == 2

    def test_illoc_on_pandas(self):
        df = DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [3, 4, 5], 'i1': [0, 0, 1], 'i2': [0, 1, 0]})
        df.set_index(['i1', 'i2'], inplace=True)
        assert len(df.illoc()['i1', 0]) == 2
