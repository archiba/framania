from pathlib import Path

from flask import Flask

from framania.hvmania.manager import HVManiaManager


class HVManiaServer(Flask):
    def __init__(self, import_name, name: str, directory: Path, renderer: str = 'bokeh', *args, **kwargs):
        super().__init__(import_name, *args, **kwargs)
        self.manager = HVManiaManager(name, directory, renderer)


def build_app(name: str, directory: Path, renderer: str = 'bokeh'):
    _app = HVManiaServer(__name__, name, directory, renderer)

    @_app.route('/hvplot/<name>')
    def chart(name):
        html_file = _app.manager[name]
        return html_file.open('rb').read()

    return _app
