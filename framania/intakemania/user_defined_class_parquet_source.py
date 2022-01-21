import pickle
from base64 import b64decode, b64encode
from pathlib import Path
from time import time
from typing import List, Union

import intake
import numpy
from intake.container.persist import PersistStore
from intake_parquet import ParquetSource
import dask.dataframe as dd
from pandas import Series


def make_meta(col):
    return Series(dtype=numpy.object)


# TODO(higumachan): のちのちに辛くなってくる可能性があるので可能ならばJSONで書き出したりができるようにしておきたい
def _dump_with_pickle(obj):
    s = pickle.dumps(obj)
    return b64encode(s).decode("ascii")


def _load_with_pickle(pickle_str: str):
    b = b64decode(pickle_str.encode("ascii"))
    return pickle.loads(b)


class UserDefinedClassParquetSource(ParquetSource):
    """
    originalのPython classが入っていたdask dataframeをpaquetから読み込むときに使うSource
    """
    def to_spark(self):
        raise NotImplementedError("only use with dask")

    def _to_dask(self):
        super()._to_dask()
        for col in self.metadata["user_defined_class_columns"]:
            self._df[col] = self._df[col].map(_load_with_pickle, meta=make_meta(col))
        return self._df

    @classmethod
    def _from_parqeuet_source(cls, ps: ParquetSource):
        ccps = cls.__new__(cls)
        ccps.__dict__ = ps.__dict__
        return ccps

    def persist(self, ttl=None, **kwargs):
        if self.container != 'dataframe':
            raise ValueError(f"{self.__class__.classname} dataframe only")
        store = PersistStore()
        path = store.getdir(self)
        out = self.export(path, **kwargs)
        out.metadata.update({
            "ttl": ttl,
            'cat': {} if self.cat is None else self.cat.__getstate__(),
        })
        from dask.base import tokenize
        tok = tokenize(self)
        store.add(tok, out)
        return out

    def export(self, path, **kwargs):
        df = self.to_dask()
        out = upload_with_user_defined_class(df, path, self.metadata["user_defined_class_columns"], **kwargs)
        from dask.base import tokenize
        tok = tokenize(self)
        metadata = {'timestamp': time(),
                    'original_metadata': self.metadata,
                    'original_source': self.__getstate__(),
                    'original_name': self.name,
                    'original_tok': tok,
                    'persist_kwargs': kwargs,}
        out.metadata.update(metadata)
        out.name = self.name
        return out


def upload_with_user_defined_class(df: dd.DataFrame, path: Union[str, Path], user_defined_class_columns: List[str], **kwargs) -> UserDefinedClassParquetSource:
    """
    originalのPython classが入っているdask dataframeをparquet形式でintakeにuploadするときに使う関数。
    intake.uploadの代わりに利用することができます。
    :param df: 
    :param path: 
    :param user_defined_class_columns: originalのPython classが入っている列の名前のリスト
    :param kwargs: pass intake.upload
    :return: 
    """
    df = df.copy()
    for col in user_defined_class_columns:
        df[col] = df[col].map(_dump_with_pickle, meta=make_meta(col))
    parquet_source: ParquetSource = intake.upload(df, str(path), **kwargs)
    parquet_source.metadata.update({"user_defined_class_columns": user_defined_class_columns})
    return UserDefinedClassParquetSource._from_parqeuet_source(parquet_source)
