import json
from pathlib import Path

import holoviews.plotting.renderer


class HVManiaManager():
    def __init__(self, name: str, directory: Path, renderer: str = 'bokeh'):
        self.name = name
        self.directory = directory

        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)

        if not self.directory.is_dir():
            raise Exception(f'{self.directory} is not directory.')

        self.catalog_file = self.directory / f'{name}.json'
        if not self.catalog_file.exists():
            self.dump_catalog({})

        self.renderer = renderer

    def load_catalog(self):
        catalog = json.load(self.catalog_file.open('r'), encoding='utf-8')
        return catalog

    def dump_catalog(self, catalog):
        json.dump(catalog, self.catalog_file.open('w'), ensure_ascii=False, indent=4, sort_keys=True)

    def __getitem__(self, item: str) -> Path:
        return Path(self.load_catalog()[item]['path'])

    def __setitem__(self, key: str, value):
        p = self.directory / f'{key}.html'
        holoviews.save(value, filename=str(p), backend=self.renderer)
        c = self.load_catalog()
        c[key] = {'path': str(p.absolute())}
        self.dump_catalog(c)
