"""Isolated evidence-ID recovery; preserve the original calibration and its failures."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from drama_local.runtime import Context, atomic_json
from tools import scene_gate_calibration as original
from service.drama_by_creativity import scene_state_gate as gate

ROOT = REPO / 'output/scene_gate_multitype_recovery_v1'
SOURCE = REPO / 'output/scene_gate_multitype_v1'
EVIDENCE_FIELDS = ('情节概要', '主要情节', '冲突与张力')
RULES = gate.RULES.split('只输出JSON：', 1)[0] + '''
引用协议：scene_evidence_catalog 是从候选场次中机械提取的原文字段，每段有 S 编号及场次编号。
引用时只选择真实的 scene_span_id 和 state_ids，不自行抄写或改写引文。角色名单不是动作证据。
只输出JSON：{"status":"pass/fail/uncertain","violations":[{"category":"life/object/resource/knowledge/location/relationship/other","state_ids":["M0001"],"scene_span_id":"S0001","reason":"简短说明矛盾"}],"reason":"整体判断"}。
fail至少有一条有证据的冲突，pass必须没有冲突；uncertain说明不确定依据。没有可引用的已证实冲突时不要虚构编号或为了完成格式而改判pass。
不要输出scene_quote或scene_id，程序会按编号取回准确的原文和所属场次。最多列6条代表性冲突，每项reason尽量不超过120字，不输出反复推演过程。
格式校验失败不等于事实判断失败，根据错误修正结构和引用，不为通过校验而改变事实判断。'''


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependencies():
    return {**original.dependencies(), 'recovery': file_hash(Path(__file__))}


def evidence_catalog(scenes):
    catalog = {}
    for scene in scenes:
        for field in EVIDENCE_FIELDS:
            for text in gate.text_values(scene.get(field, [])):
                if text.strip():
                    catalog[f'S{len(catalog) + 1:04d}'] = {
                        'scene_id': scene['场次编号'], 'field': field, 'text': text}
    if not catalog:
        raise ValueError('No scene action evidence available')
    return catalog


def validate_indexed(raw, scenes, states):
    value = json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
    if not isinstance(value, dict) or not isinstance(value.get('violations'), list):
        raise ValueError('Require object with violations list')
    spans = evidence_catalog(scenes)
    for index, item in enumerate(value['violations']):
        if not isinstance(item, dict):
            raise ValueError(f'violations[{index}] must be an object')
        span_id = item.get('scene_span_id')
        if not isinstance(span_id, str) or span_id not in spans:
            raise ValueError(f'violations[{index}] must select an existing scene_span_id from {list(spans)}')
        span = spans[span_id]
        for field, expected in [('scene_id', span['scene_id']), ('scene_quote', span['text'])]:
            if field in item and item[field] != expected:
                raise ValueError(f'violations[{index}].{field} conflicts with selected span; omit it')
            item[field] = expected
    return gate.validate_check(json.dumps(value, ensure_ascii=False), scenes, states)


class IndexedGate(gate.SceneStateGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binding['evidence_protocol'] = 'scene_span_ids_v1'
        self.binding['recovery_implementation'] = dependencies()

    def packet(self, scenes):
        return {**super().packet(scenes), 'scene_evidence_catalog': evidence_catalog(scenes)}

    async def check(self, ctx, scenes, slot='check_before', model='Qwen3.6-27B', base_url=None):
        return await self.request(ctx, slot, RULES, json.dumps(self.packet(scenes), ensure_ascii=False),
                                  lambda raw: validate_indexed(raw, scenes, self.catalog), model, base_url)

    def verified_check(self, slot, scenes, model):
        path = self.directory / f'{slot}.json'
        cached = gate.read(path)
        request = cached['request']
        if (cached['fingerprint'] != gate.digest(request) or request['binding'] != self.binding or
                request['model'] != model or request['system'] != RULES or
                request['user'] != json.dumps(self.packet(scenes), ensure_ascii=False)):
            raise ValueError(f'Indexed check input changed: {path}')
        checked = validate_indexed(cached['raw'], scenes, self.catalog)
        if checked != cached['result']:
            raise ValueError(f'Indexed check result changed: {path}')
        return checked

    def read_outcome(self):
        scenes = self.saved_scenes()
        result = gate.read(self.directory / 'outcome.json')
        if (scenes is None or result['binding_sha256'] != gate.digest(self.binding) or
                result['original_scenes_sha256'] != gate.digest(scenes) or
                result['scenes_sha256'] != gate.digest(result['scenes'])):
            raise ValueError('Recovery outcome/input binding changed')
        gate.validate_scene_outline(json.dumps(result['scenes'], ensure_ascii=False), self.episode)
        before = self.verified_check('check_before', scenes, 'Qwen3.6-27B')
        after = self.verified_check('check_after', result['scenes'], 'Qwen3.6-27B') if result['repair_attempted'] else before
        if (result['before'] != before or result['after'] != after or
                result['allowed'] != (after['status'] == 'pass')):
            raise ValueError('Recovery outcome disagrees with checks')
        return result


def prepare(source, root):
    if (root / 'manifest.json').exists():
        load(root)
        return
    if root == source or (root.exists() and any(root.iterdir())):
        raise ValueError('Use a fresh recovery directory, not the original experiment')
    _, cases = original.load(source)
    tasks, files = [], [source / 'manifest.json', source / 'cases.json']
    for case in cases:
        for run in original.RUNS:
            instance = original.make_gate(source, case, run)
            complete = (instance.directory / 'outcome.json').exists()
            if complete:
                original.read_outcome(instance)
                files.extend(path for path in instance.directory.glob('*.json')
                             if not path.name.startswith('judge_'))
            tasks.append({'case': case['id'], 'run': run, 'source_complete': complete})
    atomic_json(root / 'cases.json', cases)
    atomic_json(root / 'manifest.json', {
        'source': str(source), 'source_files': {str(path): file_hash(path) for path in files},
        'cases_sha256': gate.digest(cases), 'implementation': dependencies(), 'tasks': tasks,
        'scope': 'Mixed protocol diagnostic recovery, not a uniform efficacy experiment; original successful results retained.'})


def load(root):
    manifest, cases = gate.read(root / 'manifest.json'), gate.read(root / 'cases.json')
    if manifest['implementation'] != dependencies() or manifest['cases_sha256'] != gate.digest(cases):
        raise ValueError('Recovery code or cases changed; use a fresh directory')
    for name, expected in manifest['source_files'].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f'Frozen original result changed: {name}')
    original.load(Path(manifest['source']))
    for task in manifest['tasks']:
        if not task['source_complete']:
            path = Path(manifest['source']) / 'cases' / task['case'] / task['run'] / 'outcome.json'
            if path.exists():
                raise ValueError('Original pending task was rerun after recovery preparation; use a fresh recovery directory')
    return manifest, cases


def make_gate(root, case, run, subdirectory='cases'):
    return IndexedGate(root / subdirectory / case['id'] / run, case['episode'], case['params'], case['view'], 'enforce')


def outcome_for(root, manifest, case, task):
    if task['source_complete']:
        return original.read_outcome(original.make_gate(Path(manifest['source']), case, task['run']))
    return make_gate(root, case, task['run']).read_outcome()


def judge_for(root, case, task, outcome):
    instance = make_gate(root, case, task['run'], 'judges')
    instance.binding['generation_outcome_sha256'] = gate.digest(outcome)
    return instance


def report(root):
    manifest, cases = load(root)
    lookup = {case['id']: case for case in cases}
    rows = []
    for task in manifest['tasks']:
        case = lookup[task['case']]
        row = {**task, 'expected': case['expected'], 'origin': case['origin'], 'complete': False, 'scored': False,
               'protocol': 'original_quotes' if task['source_complete'] else 'recovery_span_ids'}
        if task['source_complete'] or (make_gate(root, case, task['run']).directory / 'outcome.json').exists():
            result = outcome_for(root, manifest, case, task)
            row.update(complete=True, before=result['before']['status'], after=result['after']['status'],
                       allowed=result['allowed'], repaired=result['repair_attempted'])
            judge = judge_for(root, case, task, result)
            for phase, scenes in [('before', case['scenes']), ('after', result['scenes'])]:
                if (judge.directory / f'judge_{phase}.json').exists():
                    row[f'judge_{phase}'] = judge.verified_check(f'judge_{phase}', scenes, 'Qwen3.8-27B')['status']
            row['scored'] = 'judge_before' in row and 'judge_after' in row
        rows.append(row)
    retained = sum(row['source_complete'] for row in rows)
    recovered = sum(row['complete'] and not row['source_complete'] for row in rows)
    print(f"original_retained={retained} recovered={recovered}/{len(rows) - retained} "
          f"complete={retained + recovered}/{len(rows)} scored={sum(row['scored'] for row in rows)}/{len(rows)}")
    for protocol in ('original_quotes', 'recovery_span_ids'):
        for expected in ('fail', 'pass', 'uncertain'):
            group = [row for row in rows if row['protocol'] == protocol and row['expected'] == expected]
            if group:
                print(json.dumps({'protocol': protocol, 'expected': expected, 'total': len(group),
                                  'complete': sum(row['complete'] for row in group),
                                  'detected_as_expected': sum(row.get('before') == expected for row in group),
                                  'blocked': sum(row['complete'] and not row['allowed'] for row in group)}, ensure_ascii=False))
    return {'scope': manifest['scope'], 'results': rows}


async def run_task(root, manifest, case, task, evaluate):
    instance = make_gate(root, case, task['run'])
    if evaluate:
        outcome = outcome_for(root, manifest, case, task)
        instance = judge_for(root, case, task, outcome)
    instance.directory.mkdir(parents=True, exist_ok=True)
    with (instance.directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if evaluate:
            url = os.environ.get('DRAMA_EVAL_BASE_URL', 'http://127.0.0.1:8001/v1')
            await instance.check(Context(), case['scenes'], 'judge_before', 'Qwen3.8-27B', url)
            await instance.check(Context(), outcome['scenes'], 'judge_after', 'Qwen3.8-27B', url)
        else:
            if task['source_complete']:
                raise ValueError('Refusing to regenerate successful original task')
            try:
                await instance.apply(Context(), case['scenes'])
            except gate.SceneGateBlocked:
                pass
    return case['id'] + '/' + task['run']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'evaluate', 'status', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-dir', type=Path, default=SOURCE)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--execute-api', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('--workers must be between 1 and 4')
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(args.source_dir.resolve(), root)
        print(f'Prepared {root}; no API calls')
        return
    manifest, cases = load(root)
    summary = report(root)
    if args.action in {'status', 'report'}:
        if args.action == 'report':
            atomic_json(root / 'summary.json', summary)
        return
    evaluate = args.action == 'evaluate'
    if evaluate and any(not row['complete'] for row in summary['results']):
        raise ValueError('Finish recovery generation before evaluation')
    pending = {(row['case'], row['run']) for row in summary['results'] if not row['scored' if evaluate else 'complete']}
    tasks = [task for task in manifest['tasks'] if (task['case'], task['run']) in pending]
    print(f'pending={len(tasks)} stage={args.action}')
    if not args.execute_api or not tasks:
        print('No API calls; add --execute-api if tasks are pending')
        return
    model, url = ('Qwen3.8-27B', os.environ.get('DRAMA_EVAL_BASE_URL', 'http://127.0.0.1:8001/v1')) if evaluate else ('Qwen3.6-27B', os.environ.get('DRAMA_LLM_BASE_URL', 'http://127.0.0.1:8000/v1'))
    original.base.verify_server(url, model)
    lookup = {case['id']: case for case in cases}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(asyncio.run, run_task(root, manifest, lookup[task['case']], task, evaluate)): task for task in tasks}
        for future in as_completed(futures):
            try:
                print('completed=' + future.result(), flush=True)
            except Exception as error:
                failures.append(futures[future])
                print(json.dumps({'task': futures[future], 'error': str(error)}, ensure_ascii=False), flush=True)
    report(root)
    if failures:
        raise SystemExit('Some recovery requests failed; valid stages retained; rerun this recovery command')


if __name__ == '__main__':
    main()
