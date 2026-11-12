"""These tests run the public commands end to end, including writing and resuming live episodes."""
import json
import subprocess
import sys

from diagnose.model import read_jsonl


def command(*args):
    result = subprocess.run([sys.executable, '-m', 'diagnose.cli', *map(str, args)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


def test_bank_command_and_manifest(tmp_path):
    path = tmp_path / 'bank.json'
    command('bank', '--count', 4, '--horizon', 1, '--max-episodes', 10, '--output', path)
    bank = json.loads(path.read_text())
    assert bank['manifest']['actual'] == len(bank['situations']) == 4
    assert {r['split'] for r in bank['situations']} == {'train', 'test'}


def test_baseline_live_command_resume_and_condition_guard(tmp_path):
    path = tmp_path / 'live.jsonl'
    args = ['live', '--baseline', 'advance', '--episodes', 2, '--output', path]
    command(*args)
    before = path.read_text()
    command(*args)
    assert path.read_text() == before
    assert len(read_jsonl(path)) == 2
    bad = subprocess.run([sys.executable, '-m', 'diagnose.cli', 'live', '--baseline', 'stay',
                          '--output', str(path)], capture_output=True, text=True)
    assert bad.returncode != 0 and 'another condition' in bad.stderr


def test_summary_command_handles_empty_results(tmp_path):
    path = tmp_path / 'responses.jsonl'
    path.write_text('')
    assert json.loads(command('summarize', path).stdout) == []
