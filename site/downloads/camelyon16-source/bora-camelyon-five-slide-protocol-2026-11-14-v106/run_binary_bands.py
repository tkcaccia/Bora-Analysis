"""One fail-fast Chiamaka GPU bundle for one frozen CAM method and test slide."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
MIG = ROOT / 'migration'
SOURCE = MIG / 'bora-camelyon-five-slide-protocol-2026-11-14-v103/infer_binary_band.py'
INPUTS = SOURCE.with_name('INPUTS_FROZEN.json')
OUTPUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v103/binary-network-bands'
PYTHON = DATA / 'experiments/virchow2-inverse-2026-10-16/.venv/bin/python'
GPU_UUID = 'GPU-59b2b884-a1c9-f7a3-9f35-e8380f4609d5'
CAP = 200 * (1 << 30)
RESERVE = 100 * (1 << 30)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True,
                          timeout=10).stdout.strip()


def compute_contexts():
    rows = command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                    '--format=csv,noheader,nounits']).splitlines()
    result = []
    for row in rows:
        if not row.strip():
            continue
        pid, gpu = [x.strip() for x in row.split(',', 1)]
        result.append({'pid': int(pid), 'gpu_uuid': gpu})
    return result


def service_cgroup(unit):
    raw = command(['systemctl', '--user', 'show', unit, '-p', 'ControlGroup'])
    value = raw.partition('=')[2]
    if not value:
        raise RuntimeError('Controller service cgroup is unavailable')
    return Path('/sys/fs/cgroup' + value)


def owned_pids(cgroup):
    return {int(x) for x in (cgroup / 'cgroup.procs').read_text().split()}


def context_ownership(contexts, cgroup):
    owned, unrelated, unknown = [], [], []
    group = str(cgroup).removeprefix('/sys/fs/cgroup')
    for item in contexts:
        pid = item['pid']
        path = Path('/proc') / str(pid)
        try:
            cgroups = (path / 'cgroup').read_text()
            name = (path / 'comm').read_text().strip().lower()
        except OSError:
            unknown.append(item)
            continue
        if group in cgroups:
            owned.append(item)
        elif 'rustdesk' in name:
            unrelated.append(item)
        else:
            unknown.append(item)
    return {'owned': owned, 'unrelated_ui': unrelated, 'unknown': unknown}


def running_bora(exclude):
    rows = command(['systemctl', '--user', 'list-units', 'bora-*',
                    '--state=running', '--no-legend', '--no-pager'])
    return [row.split()[0] for row in rows.splitlines() if row.strip()
            and row.split()[0] != exclude]


def pgid_pids(group):
    found = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().split(') ', 1)[1].split()
            if int(fields[2]) == group:
                found.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            pass
    return found


def cleanup(group, cgroup):
    observations = []
    for _ in range(11):
        pids = owned_pids(cgroup) - {os.getpid()}
        group_pids = pgid_pids(group)
        gpu = context_ownership(compute_contexts(), cgroup)
        row = {'utc': now(), 'owned_cgroup_pids': sorted(pids),
               'owned_process_group_pids': group_pids, 'gpu': gpu}
        observations.append(row)
        if not pids and not group_pids and not gpu['owned'] and not gpu['unknown']:
            return {'status': 'VERIFIED', 'observations': observations}
        time.sleep(3)
    return {'status': 'UNKNOWN_OR_FAILED', 'observations': observations}


def free_bytes():
    stat = os.statvfs(DATA)
    return stat.f_bavail * stat.f_frsize


def starts(length):
    if length <= 512:
        return [0]
    values = list(range(0, length - 512 + 1, 384))
    if values[-1] != length - 512:
        values.append(length - 512)
    return values


def verify_band(method, slide, band, state, frozen):
    folder = OUTPUT / method / slide / f'band-{band:04d}'
    receipt_path = folder / 'BAND_COMPLETE.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt['status'], receipt['method'], receipt['slide'], receipt['band']) != (
            'CAM_BINARY_NETWORK_SOURCE_ONLY_BAND_COMPLETE', method, slide, band):
        raise ValueError('Binary band receipt identity differs')
    if (receipt['input_freeze_sha256'] != sha(INPUTS) or
            receipt['source_sha256'] != sha(SOURCE) or
            receipt['checkpoint_sha256'] != frozen['model_checkpoint_sha256'] or
            receipt['expert_reference_opened']):
        raise ValueError('Binary band model, source or reference contract differs')
    for key in ('score_path', 'count_path'):
        path = Path(receipt[key])
        if sha(path) != receipt[key.replace('_path', '_sha256')]:
            raise ValueError('Binary band output hash differs')
        state['new_output_bytes'] += path.stat().st_size
    state['new_output_bytes'] += receipt_path.stat().st_size
    if state['new_output_bytes'] > frozen['new_output_bytes_bound']:
        raise OSError('Frozen binary-band output bound exceeded')
    state['completed'].append({'method': method, 'slide': slide, 'band': band,
                               'receipt_sha256': sha(receipt_path)})


def one(method, slide, band, state, frozen, cgroup, run_dir):
    service = frozen['service']
    if running_bora(service):
        raise RuntimeError('Overlapping Bora service before next band')
    gpu = context_ownership(compute_contexts(), cgroup)
    if gpu['owned'] or gpu['unknown']:
        raise RuntimeError('Owned/unknown GPU context before next band')
    if free_bytes() < RESERVE + 2 * (1 << 30):
        raise OSError('100-GiB reserve plus bounded band margin unavailable')
    if state['campaign_bytes_before'] + state['new_output_bytes'] + 2 * (1 << 30) > CAP:
        raise OSError('200-GiB external campaign cap would be exceeded')
    folder = OUTPUT / method / slide / f'band-{band:04d}'
    if folder.exists():
        raise FileExistsError('Existing band cannot be retried in place')
    log = run_dir / 'logs' / f'band-{band:04d}.log'
    with log.open('x') as stream:
        process = subprocess.Popen([str(PYTHON), str(SOURCE), '--method', method,
                                    '--slide', slide, '--band', str(band)],
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True,
                                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0',
                                        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
        state['active'] = {'band': band, 'pid': process.pid, 'pgid': process.pid,
                           'log': str(log)}
        atomic(run_dir / 'CONTROLLER_STATE.json', state)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            code = None
    proof = cleanup(process.pid, cgroup)
    state['latest_cleanup'] = proof
    atomic(run_dir / 'CONTROLLER_STATE.json', state)
    if proof['status'] != 'VERIFIED':
        raise RuntimeError('Owned/GPU cleanup unknown after band: ' + str(log))
    if code is None:
        raise TimeoutError('Binary GPU band timed out: ' + str(log))
    if code:
        raise RuntimeError(f'Binary GPU band exited {code}: {log}')
    verify_band(method, slide, band, state, frozen)
    state['active'] = None
    atomic(run_dir / 'CONTROLLER_STATE.json', state)


def main(run_dir):
    run_dir = Path(run_dir)
    full_path = run_dir / 'FULL_RUN_FROZEN.json'
    frozen = json.loads(full_path.read_text())
    method, slide = frozen['method'], frozen['slide']
    service = frozen['service']
    state_path = run_dir / 'CONTROLLER_STATE.json'
    if state_path.exists():
        raise FileExistsError('No in-place GPU bundle retry')
    if (sha(__file__) != frozen['controller_sha256'] or
            sha(SOURCE) != frozen['band_source_sha256'] or
            sha(INPUTS) != frozen['band_inputs_sha256'] or
            sha(Path(frozen['pilot_audit_path'])) != frozen['pilot_audit_sha256']):
        raise ValueError('Frozen source, weights or pilot audit changed')
    if socket.gethostname() != 'icgeb-bioinformatics-unit':
        raise RuntimeError('Chiamaka-only GPU bundle')
    if command(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']) != GPU_UUID:
        raise RuntimeError('Assigned GPU identity differs')
    cgroup = service_cgroup(service)
    if running_bora(service):
        raise RuntimeError('Overlapping Bora service at admission')
    gpu = context_ownership(compute_contexts(), cgroup)
    if gpu['owned'] or gpu['unknown']:
        raise RuntimeError('Owned or unknown GPU context at admission')
    if (frozen['campaign_bytes_before'] + frozen['new_output_bytes_bound'] > CAP or
            free_bytes() < RESERVE + frozen['new_output_bytes_bound']):
        raise OSError('Campaign cap or 100-GiB free reserve unavailable')
    inputs = json.loads(INPUTS.read_text())
    if (inputs['models'][method]['checkpoint_sha256'] != frozen['model_checkpoint_sha256'] or
            sha(Path(inputs['models'][method]['checkpoint_path'])) != frozen['model_checkpoint_sha256']):
        raise ValueError('Pair-fitted checkpoint identity differs')
    if method not in ('unet', 'deeplabv3plus') or slide not in inputs['slides']:
        raise ValueError('Method/slide outside frozen three-test study')
    groups = (len(starts(inputs['slides'][slide]['calibrated_shape_yx'][0])) +
              inputs['patch_rows_per_band'] - 1) // inputs['patch_rows_per_band']
    (run_dir / 'logs').mkdir(exist_ok=False)
    state = {'status': 'RUNNING', 'start_utc': now(), 'method': method, 'slide': slide,
             'service': service, 'assigned_gpu_uuid': GPU_UUID, 'host': socket.gethostname(),
             'frozen_sha256': sha(full_path), 'campaign_bytes_before': frozen['campaign_bytes_before'],
             'free_bytes_at_admission': free_bytes(), 'new_output_bytes': 0,
             'completed': [], 'latest_cleanup': None, 'active': None,
             'test_reference_opened': False, 'online_publication': False}
    atomic(state_path, state)
    deadline = time.monotonic() + frozen['internal_deadline_seconds']
    try:
        for band in range(groups):
            if band == frozen.get('reused_pilot_band'):
                verify_band(method, slide, band, state, frozen)
                state['completed'][-1]['reused_audited_pilot'] = True
                atomic(state_path, state)
                continue
            if time.monotonic() + 720 > deadline:
                raise TimeoutError('Stop admitting bands before bundle deadline')
            one(method, slide, band, state, frozen, cgroup, run_dir)
        gpu = context_ownership(compute_contexts(), cgroup)
        if running_bora(service) or gpu['owned'] or gpu['unknown']:
            raise RuntimeError('Unexpected service or GPU context at terminal state')
        state.update(status='COMPLETE', end_utc=now())
        atomic(state_path, state)
        atomic(run_dir / 'BANDS_COMPLETE.json',
               {'status': 'BINARY_NETWORK_SOURCE_ONLY_BANDS_COMPLETE',
                'method': method, 'slide': slide, 'bands': len(state['completed']),
                'controller_state_sha256': sha(state_path),
                'test_reference_opened': False, 'online_publication': False})
    except BaseException as error:
        active = state.get('active')
        if active and (state.get('latest_cleanup') or {}).get('status') != 'VERIFIED':
            state['latest_cleanup'] = cleanup(active['pgid'], cgroup)
        state.update(status='FAILED_LATCHED_PRESERVE_PARTIAL', error=repr(error), end_utc=now())
        atomic(state_path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    main(args.run_dir)
