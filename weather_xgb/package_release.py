"""Package a relocatable English Windows distribution and dependency notices."""
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import json
import shutil
import sys

from .config import dump
from .examples import generate


def example_project(folder):
    config = asdict(generate(folder))
    for spec in config['variables'] + config['backgrounds'] + config['shapes']:
        spec['source']['path'] = Path(spec['source']['path']).name
    config['output_dir'] = '../runs'
    dump(config, Path(folder) / 'example_project.json')


def package():
    root = Path(__file__).resolve().parent.parent
    release = root / 'release' / 'WeatherXGB-English'
    if not (release / 'WeatherXGB-English.exe').is_file():
        raise FileNotFoundError('Build WeatherXGB-English.spec before packaging.')
    example_project(root / 'examples')
    shutil.copytree(root / 'examples', release / 'examples', dirs_exist_ok=True)
    for name in ('README.md', 'requirements.txt', 'requirements-lock.txt', 'THIRD_PARTY_NOTICES.md'):
        shutil.copy2(root / name, release / name)
    shutil.copytree(root / 'docs', release / 'docs', dirs_exist_ok=True)
    site = Path(sys.prefix) / 'Lib' / 'site-packages'
    notices = release / 'third_party_licenses'
    notices.mkdir(exist_ok=True)
    for metadata in site.glob('*.dist-info'):
        for source in metadata.rglob('*'):
            if source.is_file() and (source.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')) or 'licenses' in source.parts):
                target = notices / metadata.name / source.relative_to(metadata)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    versions = sorted((dict(name=d.metadata['Name'], version=d.version) for d in importlib.metadata.distributions()), key=lambda d: d['name'].lower())
    dump(versions, release / 'dependency_versions.json')
    (release / 'Open synthetic example.bat').write_text('@echo off\ncd /d "%~dp0"\nstart "" "WeatherXGB-English.exe" --project "examples\\example_project.json"\n', encoding='ascii')
    print(release)
    return release


if __name__ == '__main__':
    package()
