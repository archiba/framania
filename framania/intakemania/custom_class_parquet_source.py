import pickle
from base64 import b64decode, b64encode
from pathlib import Path
from typing import List, Union

import intake
from intake_parquet import ParquetSource
import dask.dataframe as dd


# TODO(higumachan): のちのちに辛くなってくる可能性があるので可能ならばJSONで書き出したりができるようにしておきたい
def _dump_with_pickle(obj):
    s = pickle.dumps(obj)
    return b64encode(s).decode("ascii")


def _load_with_pickle(pickle_str: str):
    b = b64decode(pickle_str.encode("ascii"))
    return pickle.loads(b)


class CustomClassParquetSource(ParquetSource):
    """
    originalのPython classが入っていたdask dataframeをpaquetから読み込むときに使うSource
    """
    def to_spark(self):
        raise NotImplementedError("only use with dask")

    def _to_dask(self):
        super()._to_dask()
        for col in self.metadata["custom_class_columns"]:
            self._df[col] = self._df[col].map(_load_with_pickle, meta=dd.utils.make_meta(self._df[col]))
        return self._df

    @classmethod
    def _from_parqeuet_source(cls, ps: ParquetSource):
        ccps = cls.__new__(cls)
        ccps.__dict__ = ps.__dict__
        return ccps


def upload_with_custom_class(df: dd.DataFrame, path: Union[str, Path], custom_class_columns: List[str], **kwargs) -> CustomClassParquetSource:
    """
    オリジナルのPython classが入っているdask dataframeをparquet形式でintakeにuploadするときに使う関数。
    intake.uploadの代わりに利用することができます。
    :param df: 
    :param path: 
    :param custom_class_columns: originalのPython classが入っている列の名前のリスト
    :param kwargs: pass intake.upload
    :return: 
    """
    df = df.copy()
    for col in custom_class_columns:
        df[col] = df[col].map(_dump_with_pickle, meta=dd.utils.make_meta(df[col]))
    parquet_source: ParquetSource = intake.upload(df, str(path), **kwargs)
    parquet_source.metadata.update({"custom_class_columns": custom_class_columns})
    return CustomClassParquetSource._from_parqeuet_source(parquet_source)
