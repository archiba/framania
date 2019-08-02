import os
from pathlib import Path

from IPython import get_ipython
from IPython.core.magic import register_line_magic

from build.lib.framania.hvmania.manager import HVManiaManager
from framania.intakemania.extension import get_version_name
from framania.intakemania.extension import FramaniaExtendedIntakeCatalog


@register_line_magic
def start_nb(line):
    parts = line.split(' ')
    if len(parts) != 3:
        print('Make sure input `%start_nb {project title} {catalog file path} {data directory path}`')

    analysis_name, catalog_file, data_dir = parts
    
    get_ipython().user_global_ns['__framania_pj_name__'] = analysis_name
    get_ipython().user_global_ns['__framania_catalog_file__'] = Path(catalog_file)
    get_ipython().user_global_ns['__framania_data_dir__'] = Path(data_dir)

    print(f'Start analysis notebook {analysis_name} (data directory: {Path(data_dir)})')


@register_line_magic
def begin_analysis(line):

    if get_ipython().user_global_ns.get('__framania_analysis_finished__', True) is False:
        print('Make sure previous analysis is ended.')
        return

    parts = line.split(' ')
    if len(parts) < 2:
        print(
            'Make sure input `%begin_analysis {analysis name} {analysis version} {upstream analysis name1} {upstream analysis name2} ...`')
    name, version = parts[:2]
    upstream_names = parts[2:]
    get_ipython().user_global_ns['__framania_analysis_name__'] = name
    get_ipython().user_global_ns['__framania_analysis_version__'] = version
    get_ipython().user_global_ns['__framania_analysis_upstream_names__'] = upstream_names
    get_ipython().user_global_ns['__framania_analysis_finished__'] = False

    catalog = FramaniaExtendedIntakeCatalog(get_ipython().user_global_ns['__framania_catalog_file__'])
    upstreams = []
    for upstream_name in upstream_names:
        try:
            upstream = catalog.find_by_version_name(upstream_name)
        except:
            upstream = catalog.find_latest_source_by_name(upstream_name)
        upstreams.append(upstream)
    get_ipython().user_global_ns['__framania_analysis_upstreams__'] = upstreams

    for upstream in upstreams:
        print(f'Register global variable {upstream.name} from {upstream.name} {upstream.version}.')
        get_ipython().user_global_ns[upstream.name] = upstream.intake_source.to_dask()


@register_line_magic
def end_analysis(line):
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

    print('End analysis.')
    get_ipython().user_global_ns['__framania_analysis_finished__'] = True

@register_line_magic
def serve_chart(line):
    pj_name = get_ipython().user_global_ns['__framania_pj_name__']
    name = get_ipython().user_global_ns['__framania_analysis_name__']
    version = get_ipython().user_global_ns['__framania_analysis_version__']
    data_dir = get_ipython().user_global_ns['__framania_data_dir__']

    result_var = get_ipython().user_global_ns[line]

    chart_name = get_version_name(name, version)
    chart_dir = data_dir / 'chart'

    manager = HVManiaManager(pj_name, chart_dir)
    manager[chart_name] = result_var

    del get_ipython().user_global_ns['__framania_analysis_name__']
    del get_ipython().user_global_ns['__framania_analysis_version__']
    del get_ipython().user_global_ns['__framania_analysis_upstream_names__']

    print('End analysis.')
    print(f'Chart is served as {chart_name}')
    get_ipython().user_global_ns['__framania_analysis_finished__'] = True

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
