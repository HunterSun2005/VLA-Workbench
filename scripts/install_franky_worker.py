#!/usr/bin/python3
"""Install the hash-locked wheel into a new isolated worker venv; never open hardware."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vla.core.franky_runtime import check_worker, worker_environment, semantic_binding


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock_path = ROOT/'scripts/franky-worker.lock.json'
    lock = json.loads(lock_path.read_text())
    if sys.version_info[:2] != (3, 10):
        raise SystemExit('Run this installer with /usr/bin/python3 (Python 3.10)')
    base = ROOT/'runtime/franky'
    base.mkdir(parents=True, exist_ok=True)
    # Keep a stable inode even after failure; this is a persistent flock file.
    with (base/'install.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        digest = sha256(lock_path)
        target = base/'envs'/digest
        python = target/'bin/python'
        marker = target/'installation.json'
        if target.exists() and not marker.is_file():
            raise RuntimeError(f'Incomplete environment preserved for diagnosis: {target}')
        if not target.exists():
            wheels = base/'wheels'
            wheels.mkdir(exist_ok=True)
            for item in lock['wheels']:
                path = wheels/item['filename']
                if not path.exists():
                    partial = path.with_suffix('.download')
                    with urllib.request.urlopen(item['url'], timeout=120) as response, partial.open('wb') as out:
                        import shutil
                        shutil.copyfileobj(response, out)
                    if sha256(partial) != item['sha256']:
                        raise RuntimeError(f'Wheel hash mismatch: {partial}')
                    partial.replace(path)
                if sha256(path) != item['sha256']:
                    raise RuntimeError(f'Wheel hash mismatch: {path}')
            requirements = base/f'{digest}.requirements.txt'
            requirements.write_text(''.join(f"{a['name']}=={a['version']} --hash=sha256:{a['sha256']}\n"
                                             for a in lock['wheels']))
            env = worker_environment(ROOT, '/usr/bin/python3')
            subprocess.run(['/usr/bin/python3', '-m', 'venv', '--copies', '--without-pip', str(target)],
                           env=env, check=True)
            site = target/'lib/python3.10/site-packages'
            subprocess.run(['/usr/bin/python3', '-m', 'pip', '--isolated', 'install', '--no-deps',
                            '--no-index', '--only-binary=:all:', '--require-hashes', '--find-links', str(wheels),
                            '--target', str(site), '-r', str(requirements)], env=env, check=True)
            _, fingerprint = check_worker(ROOT, python)
            marker.write_text(json.dumps({'lock_sha256': digest, 'sources': lock,
                                          'fingerprint': fingerprint}, indent=2)+'\n')
        else:
            installed = json.loads(marker.read_text())
            if installed['lock_sha256'] != digest:
                raise RuntimeError('Environment installation marker mismatch')
            _, fingerprint = check_worker(ROOT, python)
            if semantic_binding(fingerprint) != semantic_binding(installed['fingerprint']):
                raise RuntimeError('Installed worker fingerprint changed; environment preserved for diagnosis')
        candidate = base/f'.current-{os.getpid()}'
        candidate.symlink_to(Path('envs')/digest, target_is_directory=True)
        candidate.replace(base/'current')
        print(json.dumps({'python': str(base/'current/bin/python'), 'fingerprint': fingerprint}, indent=2))


if __name__ == '__main__':
    main()
