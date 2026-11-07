import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import validate


def test_junit_preserves_skips_failures_and_errors(tmp_path):
    path = tmp_path/'model.xml'
    path.write_text('''<testsuites><testsuite>
      <testcase classname="suite" name="ok"/>
      <testcase classname="suite" name="needs_weights"><skipped message="missing PI0.5"/></testcase>
      <testcase classname="suite" name="bad"><failure message="wrong target"/></testcase>
      <testcase classname="suite" name="setup"><error message="missing dependency"/></testcase>
    </testsuite></testsuites>''')
    result = validate.finalize_check('model', {'status': 'failed', 'exit_code': 1}, tmp_path)
    assert result['counts'] == dict(passed=1, skipped=1, failed=1, errors=1)
    assert result['skipped_cases'][0]['reason'] == 'missing PI0.5'
    assert len(result['failures']) == 2 and result['status'] == 'failed'


@pytest.mark.parametrize('evidence', ['missing', 'empty', 'skipped', 'passed'])
def test_zero_exit_is_not_sufficient_test_evidence(tmp_path, evidence):
    if evidence != 'missing':
        case = {'empty': '', 'skipped': '<testcase><skipped message="assets missing"/></testcase>',
                'passed': '<testcase/>'}[evidence]
        (tmp_path/'ros.xml').write_text(f'<testsuite>{case}</testsuite>')
    result = validate.finalize_check('ros', {'status': 'passed'}, tmp_path)
    assert result['status'] == {'missing':'failed', 'empty':'failed', 'skipped':'skipped', 'passed':'passed'}[evidence]


def test_timed_out_check_retains_log_and_reports_failure(tmp_path):
    result = validate.run_check([sys.executable, '-u', '-c',
                                 "import time; print('started'); time.sleep(60)"],
        root=tmp_path, env=os.environ.copy(), log=tmp_path/'model.log', timeout_s=.2)
    assert result['status'] == 'timed_out'
    assert (tmp_path/'model.log').read_text().startswith('started\n')
    assert result['elapsed_s'] < 5


def test_report_distinguishes_unrun_checks_and_source_changes(tmp_path):
    report = {'source': {'commit':'abc', 'branch':'ros-dev', 'dirty':True, 'tree_sha256':'123'},
              'finished_at': 'today', 'checks': {
                  'model': {'status':'passed_with_skips', 'counts':dict(passed=1, skipped=1, failed=0, errors=0),
                            'skipped_cases':[{'test':'real_model', 'reason':'missing assets'}], 'log':'model.log'},
                  'timing': {'status':'not_run', 'reason':'not selected'}}}
    validate.write_report(tmp_path, report)
    saved = json.loads((tmp_path/'report.json').read_text())
    assert saved['status'] == 'passed_with_gaps'
    assert 'missing assets' in (tmp_path/'report.md').read_text()
    report['source_changed'] = True
    validate.write_report(tmp_path, report)
    assert json.loads((tmp_path/'report.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('status,frequency', [('passed', True), ('failed', False)])
def test_timing_report_is_copied_even_when_acceptance_fails(tmp_path, status, frequency):
    raw = tmp_path/'raw-report.json'
    raw.write_text(json.dumps({'frequency_target_passed': frequency, 'tracking': {'hz': 100}}))
    (tmp_path/'timing.log').write_text(f'REPORT={raw}\n')
    result = validate.finalize_check('timing', {'status':status, 'log':'timing.log'}, tmp_path)
    assert result['status'] == status
    assert json.loads((tmp_path/'timing.json').read_text())['frequency_target_passed'] is frequency


def test_source_digest_includes_untracked_edits_but_ignores_artifacts(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path/'.gitignore').write_text('artifacts/\n')
    (tmp_path/'code.py').write_text('x = 1\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], check=True)
    before = validate.source_identity(tmp_path)
    assert not before['dirty']
    (tmp_path/'artifacts').mkdir()
    (tmp_path/'artifacts/report.json').write_text('{}')
    assert validate.source_identity(tmp_path) == before
    (tmp_path/'new.py').write_text('test = 1\n')
    after = validate.source_identity(tmp_path)
    assert after['dirty'] and after['tree_sha256'] != before['tree_sha256']
    (tmp_path/'new.py').write_text('test = 2\n')
    assert validate.source_identity(tmp_path)['tree_sha256'] != after['tree_sha256']
