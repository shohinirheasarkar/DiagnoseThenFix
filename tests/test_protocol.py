"""These checks keep the research promises and package metadata readable."""
from pathlib import Path
import tomllib


def test_protocol_records_design_before_collection():
    text = Path('preregistration.md').read_text()
    for required in ('A1', 'B1', 'B2', 'B3', 'C1', 'C2', 'D1-D3', 'E1-E3',
                     'X1-X3', 'H4', '2,000', '50 episode', 'validation only',
                     'malformed', 'upper bounds', 'pending'):
        assert required in text


def test_package_installs_only_project_code():
    meta = tomllib.loads(Path('pyproject.toml').read_text())
    assert meta['tool']['setuptools']['packages']['find']['include'] == ['diagnose*']
    assert meta['project']['requires-python'] == '>=3.11'
