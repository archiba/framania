import os
from pathlib import Path
from typing import List

from IPython import get_ipython
from IPython.core.magic import register_line_magic
from intake.catalog import Catalog

from framania.hvmania.manager import HVManiaManager
from framania.intakemania.extension import get_version_name, FramaniaExtendedIntakeSource
from framania.intakemania.extension import FramaniaExtendedIntakeCatalog


@register_line_magic
def start_nb(line):
    parts = line.split(' ')
    if len(parts) != 3:
        print('Make sure input `%start_nb {project title} {catalog file path} {data directory path}`')
        return

    analysis_name, catalog_file, data_dir = parts

    get_ipython().user_global_ns['__framania_pj_name__'] = analysis_name
    get_ipython().user_global_ns['__framania_catalog_file__'] = Path(catalog_file)
    get_ipython().user_global_ns['__framania_data_dir__'] = Path(data_dir)

    print(f'Start analysis notebook {analysis_name} (data directory: {Path(data_dir)})')


def _add_analysis_upstream(catalog, upstream_names: List[str]):
    upstreams = []
    for upstream_name in upstream_names:
        try:
            upstream = catalog.find_by_version_name(upstream_name)
        except:
            upstream = catalog.find_latest_source_by_name(upstream_name)
        upstreams.append(upstream)
    get_ipython().user_global_ns['__framania_analysis_upstream_names__'].extend(upstream_names)
    get_ipython().user_global_ns['__framania_analysis_upstreams__'].extend(upstreams)

    for upstream in upstreams:
        print(f'Register upstream as a global variable {upstream.name} from {upstream.name} {upstream.version}.')
        get_ipython().user_global_ns[upstream.name] = upstream.intake_source.to_dask()


@register_line_magic
def begin_analysis(line):
    parts = line.split(' ')
    if len(parts) < 2:
        print(
            'Make sure input `%begin_analysis {analysis name} {analysis version} {upstream analysis name1} {upstream analysis name2} ...`')
        return
    name, version = parts[:2]
    upstream_names = parts[2:]
    get_ipython().user_global_ns['__framania_analysis_name__'] = name
    get_ipython().user_global_ns['__framania_analysis_version__'] = version
    get_ipython().user_global_ns['__framania_analysis_upstreams__'] = []
    get_ipython().user_global_ns['__framania_analysis_upstream_names__'] = []

    proj_name = get_ipython().user_global_ns['__framania_pj_name__']
    print(f'Begin analysis {name} in {proj_name}')
    catalog = FramaniaExtendedIntakeCatalog(get_ipython().user_global_ns['__framania_catalog_file__'])
    _add_analysis_upstream(catalog, upstream_names)


@register_line_magic
def add_analysis_upstream(line):
    if len(line) == 0:
        print('Make sure input `%add_analysis_upstream {upstream analysis name1} {upstream analysis name2} ...`')
        return
    upstream_names = line.split(' ')
    catalog = FramaniaExtendedIntakeCatalog(get_ipython().user_global_ns['__framania_catalog_file__'])
    _add_analysis_upstream(catalog, upstream_names)


@register_line_magic
def end_analysis(line):
    if '__framania_analysis_name__' not in get_ipython().user_global_ns.keys():
        print('Make sure begin analysis before end analysis.')
        return

    if len(line) >= 1:
        result_var = get_ipython().user_global_ns[line]
        catalog = FramaniaExtendedIntakeCatalog(get_ipython().user_global_ns['__framania_catalog_file__'])
        name = get_ipython().user_global_ns['__framania_analysis_name__']
        version = get_ipython().user_global_ns['__framania_analysis_version__']
        upstreams = get_ipython().user_global_ns['__framania_analysis_upstreams__']
        data_dir = get_ipython().user_global_ns['__framania_data_dir__']
        print(f'Dump dask dataframe {line} to catalog as {name}')
        catalog.dump_dask(result_var, name, version, data_dir, upstreams)

    del get_ipython().user_global_ns['__framania_analysis_name__']
    del get_ipython().user_global_ns['__framania_analysis_version__']
    del get_ipython().user_global_ns['__framania_analysis_upstream_names__']
    del get_ipython().user_global_ns['__framania_analysis_upstreams__']

    print('End analysis.')


@register_line_magic
def serve_chart(line):
    parts = line.split(' ')
    if len(parts) < 3:
        print(
            'Make sure input `%serve_chart {chart name} {chart version} {chart variable}`')
        return

    pj_name = get_ipython().user_global_ns['__framania_pj_name__']
    name = get_ipython().user_global_ns['__framania_analysis_name__']
    version = get_ipython().user_global_ns['__framania_analysis_version__']
    data_dir = get_ipython().user_global_ns['__framania_data_dir__']

    chart_name, chart_version, result_var_name = parts
    result_var = get_ipython().user_global_ns[result_var_name]

    chart_dir = data_dir / 'chart' / get_version_name(name, version)

    manager = HVManiaManager(pj_name, chart_dir)
    manager[chart_name] = result_var

    catalog = FramaniaExtendedIntakeCatalog(get_ipython().user_global_ns['__framania_catalog_file__'])
    source = FramaniaExtendedIntakeSource(
        Catalog(),
        chart_name, chart_version, None,
        get_ipython().user_global_ns['__framania_analysis_upstreams__'], False)
    catalog.append(source)

    print(f'Chart is served as {chart_name} {chart_version}')


def validation_info_text(info, nest: int, name):
    t = '\t' * nest
    texts = [f'{t}- {name} | {info[name][-1][1]}']

    for upname, upflag in info[name][:-1]:
        texts.append(validation_info_text(info, nest + 1, upname))
    return os.linesep.join(texts)


@register_line_magic
def validate_catalog(line):
    catalog = FramaniaExtendedIntakeCatalog(get_ipython().user_global_ns['__framania_catalog_file__'])
    flag, info = catalog.validate()
    print(f'Validation {flag}')
    for name in sorted(info.keys()):
        print(validation_info_text(info, 0, name))
