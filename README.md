# framania - pandas/dask DataFrame API extension library

[![Actions Status](https://github.com/archiba/framania/workflows/tests/badge.svg)](https://github.com/archiba/framania/actions)


## モチベーション

pandas/daskには、DataFrame操作のための充実したAPIが用意されています。
しかし、このAPIを使って実際に処理を行なってみると、「こんなAPIが欲しい！」という気持ちになることも多いと思います。
`framania`では、痒いところに手が届かないような、もどかしい部分のAPIを拡張することを目的に開発しています。

## コンセプト

`framania`では、下記のポイントを重要視して実装を行います。

- 高速に動作
- 省メモリ設計
- なるべくオプションは少ないAPI

## バージョン

0.11.0b4

## API一覧

- daskmania
    - aggregate
        - aggregate_by_named_index_and_keys `#`
        - aggregate_as_series `#`
        - aggregate_as_series_by_named_index_and_keys `#`
    - filter
        - drop_rows_by_index `#`
        - drop_duplicates_by_named_index_and_keys `#`
    - stack
        - stack_list_column `#`
        - stack_list_columns `#`
        - stack_dict_column `#`
        - stack_columns `#`
    - util
        - make_meta `#`
        - map_partitions_as_meta `#`
    - api `##`
- pandasmania
    - aggregate
        - aggregate_as_series
    - stack
        - stack_list_column `#`
        - stack_list_columns `#`
        - stack_dict_column `#`
        - stack_columns `#`
        - unstack_column `#`
    - na
        - fbfill_series `#`
    - filter
        - drop_duplicates_by_named_index_and_keys `#`
    - transform
        - timeseries_value_changed `#`
        - timeseries_id_changed `#`
        - grouper_for_timeseries `#`
        - group_row_number `#`
        - asstr `#`
    - api `##`
- intakemania
    - util
        - add_source_to_catalog `#`
        - dump_dask_to_intake `#`
        - dump_pandas_to_intake `#`
        - persist_local `#`
    - extension
        - FramaniaExtendedIntakeCatalog `#`
        - FramaniaExtendedIntakeSource `#`
        - analysis `#`
    - api `##`
    - jupyter

> ### testing status
> 
> - `#`: doctest
> - `##`: test file
> - `###`: doctest and test file


## テスト

テストは、doctestによる方法と、テストプログラムを作成する方法を適宜選択します。
テストデータの生成方法が複雑な場合や、様々なテストデータによるテストが必要な場合は、テストプログラムを作成します。

```bash
> pytest
```

## リリース
```bash
pip install setuptools wheel twine
python setup.py sdist
python -m twine upload --repository  testpypi dist/*
python -m twine upload --repository  pypi dist/*
rm -rf dist *.egg-info MANIFEST

```

## インストール
```bash
pip install framania
```
