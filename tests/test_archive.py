"""These checks make sure compressed research evidence survives a byte-for-byte round trip."""
import gzip
import hashlib
import json

import pytest

from scripts.package_results import package
from paper.build import rows


def test_archive_is_lossless_deterministic_and_paper_readable(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    content = b'{"correct":true}\n{"correct":false}\n'
    (source / 'responses.jsonl').write_bytes(content)
    (source / 'manifest.json').write_text('{"seed":0}')
    a, b = tmp_path / 'a', tmp_path / 'b'
    manifest = package(source, a)
    package(source, b)
    compressed = (a / 'responses.jsonl.gz').read_bytes()
    assert compressed == (b / 'responses.jsonl.gz').read_bytes()
    assert gzip.decompress(compressed) == content
    assert rows(a / 'responses.jsonl') == [{'correct': True}, {'correct': False}]
    item = next(r for r in manifest['files'] if r['source'] == 'responses.jsonl')
    assert item['source_sha256'] == hashlib.sha256(content).hexdigest()
    assert json.loads((a / 'archive.json').read_text())['lossless']
    with pytest.raises(ValueError):
        package(source, source / 'nested')
