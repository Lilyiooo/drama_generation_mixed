"""Real and explicitly controlled multi-category scene-gate probes."""

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
from service.drama_by_creativity import scene_state_gate as gate
from tools import state_constraint_branch as base

ROOT = REPO / 'output/scene_gate_multitype_v1'
SOURCE = REPO / 'output/state_constraint_branch_e45_v1'
RUNS = ('R01', 'R02', 'R03')


def dependencies():
    return {'gate': gate.implementation_hash(), 'runner': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load(root):
    manifest, cases = gate.read(root / 'manifest.json'), gate.read(root / 'cases.json')
    if manifest['cases_sha256'] != gate.digest(cases) or manifest['implementation'] != dependencies():
        raise ValueError('Calibration inputs or implementation changed; use a fresh directory')
    return manifest, cases


def scene(action):
    return {'场次编号': '2-1', '时间与地点': '密室 日', '核心功能': '取得证据并推动调查',
            '情节概要': action, '悬念钩子': '证据指向谁？', '出场人物': ['李舟', '周岚'],
            '冲突与张力': ['寻找证据时遭遇阻碍'], '主要情节': [action],
            '人物动机与目标': [{'角色': '李舟', '动机与目标': '找出调查线索'}, {'角色': '周岚', '动机与目标': '核对证据'}]}


def controlled_cases():
    groups = [
        ('life', 'character_state', '李舟在上一集明确死亡，尸体已安葬。',
         '李舟当场推开密室门，对周岚说出暗号，随后两人一起离开。',
         '周岚打开李舟生前留下的笔记，读出其中的暗号，再独自推门离开。',
         '本集开始时李舟活着，身体健康。'),
        ('object', 'resources_and_evidence', '唯一的原始账本上一集已经烧成灰烬，无法恢复；周岚保留一份此前抄录的副本。',
         '周岚直接拿出那本已经烧毁的原始账本，翻到完好的第十页当众展示。',
         '周岚说明原始账本已烧毁，取出此前抄录的副本供众人核对。',
         '本集开始时同一本唯一的原始账本完好无损，仍由周岚持有。'),
        ('resource', 'resources_and_evidence', '本集开始时周岚的钱袋中余额为零，没有其他现金或代付渠道。',
         '周岚没有筹款或借贷，直接从该钱袋取出十两银子付给守卫。',
         '周岚先当场将自己的玉簪典当获得十两银子，把新得的银子交给守卫。',
         '本集开始时同一个钱袋中有十两银子可用。'),
        ('knowledge', 'unknown_information', '本集开始时周岚不知道密室口令，从未听过口令，也没有取得任何记载口令的资料。',
         '周岚未获提示或资料，也没有猜测试探，直接凭早已掌握的口令准确打开密室。',
         '周岚先收到守卫当场交出的纸条，读出上面明确写着的口令，然后用新获知的口令打开密室。',
         '本集开始时周岚已经知道同一间密室的口令。'),
    ]
    cases = []
    for category, field, fact, bad, good, contradiction in groups:
        for variant, expected, action in [('conflict', 'fail', bad), ('legal', 'pass', good), ('ambiguous', 'uncertain', bad)]:
            view = {key: [] for key in gate.FIELDS}
            view[field] = [fact]
            if variant == 'ambiguous':
                view[field].append(contradiction)
            cases.append({'id': f'{category}_{variant}', 'origin': 'controlled_synthetic', 'category': category,
                          'expected': expected, 'episode': 2, 'view': view, 'scenes': [scene(action)],
                          'params': {'Outline': '众人在密室寻找并核对调查证据，推进调查。', 'PrevEpisodeScript': '',
                                     'EpisodeIdx': '2', 'WorldView': '架空古代侦查故事', 'RoleInfo': '李舟与周岚参与调查。'}})
    return cases


def prepare(source, root):
    if (root / 'manifest.json').exists():
        load(root)
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError('Refusing to overwrite nonempty experiment')
    parent, frozen = base.load(source)
    source_root = Path(parent['source'])
    traces = list(source_root.glob(f"diagnostics/*/model_calls/{base.TRACE_IDS['scene']}/output.txt"))
    if len(traces) != 1:
        raise ValueError('Missing original scene trace')
    original = base.validate_scene_outline(traces[0].read_text(encoding='utf-8'), 45)
    clean_path = source / 'generation/control/R01/scene.json'
    clean = gate.read(clean_path)
    if base.validate_scene_outline(clean['raw'], 45) != clean['result']:
        raise ValueError('Clean source scene cache changed')
    cases = controlled_cases()
    for name, expected, scenes in [('real_e45_conflict', 'fail', original), ('real_e45_clean', 'pass', clean['result'])]:
        cases.append({'id': name, 'origin': 'real_saved_generation', 'category': 'life', 'expected': expected,
                      'episode': 45, 'params': frozen['calls']['scene']['parameters'], 'view': frozen['selected_state'], 'scenes': scenes})
    for case in cases:
        gate.state_catalog(case['view'])
        base.validate_scene_outline(json.dumps(case['scenes'], ensure_ascii=False), case['episode'])
    sources = [source / 'inputs.json', source / 'manifest.json', traces[0], clean_path]
    atomic_json(root / 'cases.json', cases)
    atomic_json(root / 'manifest.json', {'cases_sha256': gate.digest(cases), 'implementation': dependencies(), 'runs': list(RUNS),
        'source_files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'generator': 'Qwen3.6-27B', 'judge': 'Qwen3.8-27B',
        'scope': '2 real saved scenes plus 12 controlled synthetic scenes; repeats are not independent cases; no efficacy claim for full scripts'})


def make_gate(root, case, run):
    return gate.SceneStateGate(root / 'cases' / case['id'] / run, case['episode'], case['params'], case['view'], 'enforce')


async def run_case(root, case, run, evaluate):
    instance = make_gate(root, case, run)
    directory = instance.directory
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ctx = Context()
        if evaluate:
            outcome = read_outcome(instance)
            url = os.environ.get('DRAMA_EVAL_BASE_URL', 'http://127.0.0.1:8001/v1')
            await instance.check(ctx, case['scenes'], 'judge_before', 'Qwen3.8-27B', url)
            await instance.check(ctx, outcome['scenes'], 'judge_after', 'Qwen3.8-27B', url)
        else:
            try:
                await instance.apply(ctx, case['scenes'])
            except gate.SceneGateBlocked:
                pass
    return case['id'] + '/' + run


def read_outcome(instance):
    original = instance.saved_scenes()
    result = gate.read(instance.directory / 'outcome.json')
    if original is None or result['binding_sha256'] != gate.digest(instance.binding) or result['original_scenes_sha256'] != gate.digest(original):
        raise ValueError('Outcome belongs to different inputs')
    if result['scenes_sha256'] != gate.digest(result['scenes']):
        raise ValueError('Outcome scenes changed')
    before = verified_check(instance, instance.directory / 'check_before.json', original, 'Qwen3.6-27B')
    after = verified_check(instance, instance.directory / 'check_after.json', result['scenes'], 'Qwen3.6-27B') if result['repair_attempted'] else before
    if result['before'] != before or result['after'] != after or result['allowed'] != (after['status'] == 'pass'):
        raise ValueError('Outcome does not match validated checks')
    return result


def verified_check(instance, path, scenes, model):
    cached = gate.read(path)
    request = cached['request']
    if (cached['fingerprint'] != gate.digest(request) or request['binding'] != instance.binding or
        request['model'] != model or request['system'] != gate.RULES or
        request['user'] != json.dumps(instance.packet(scenes), ensure_ascii=False)):
        raise ValueError(f'Judge input changed: {path}')
    result = gate.validate_check(cached['raw'], scenes, instance.catalog)
    if result != cached['result']:
        raise ValueError(f'Judge cache changed: {path}')
    return result


def report(root):
    manifest, cases = load(root)
    rows = []
    for case in cases:
        for run in RUNS:
            instance = make_gate(root, case, run)
            row = {'case': case['id'], 'run': run, 'origin': case['origin'], 'expected': case['expected'],
                   'category': case['category'], 'complete': False, 'scored': False}
            if (instance.directory / 'outcome.json').exists():
                outcome = read_outcome(instance)
                row.update(complete=True, before=outcome['before']['status'], after=outcome['after']['status'],
                           repaired=outcome['repair_attempted'], allowed=outcome['allowed'], changed=outcome['text_changed'])
                for phase, scenes in [('before', case['scenes']), ('after', outcome['scenes'])]:
                    path = instance.directory / f'judge_{phase}.json'
                    if path.exists():
                        checked = verified_check(instance, path, scenes, 'Qwen3.8-27B')
                        row[f'judge_{phase}'] = checked['status']
                row['scored'] = 'judge_before' in row and 'judge_after' in row
            rows.append(row)
    print(f"complete={sum(row['complete'] for row in rows)}/{len(rows)} "
          f"scored={sum(row['scored'] for row in rows)}/{len(rows)} "
          f"blocked={sum(row['complete'] and not row['allowed'] for row in rows)}")
    for origin in ('real_saved_generation', 'controlled_synthetic'):
        for expected in ('fail', 'pass', 'uncertain'):
            group = [row for row in rows if row['origin'] == origin and row['expected'] == expected]
            if group:
                print(json.dumps({'origin': origin, 'expected': expected, 'total': len(group),
                    'complete': sum(row['complete'] for row in group), 'scored': sum(row['scored'] for row in group),
                    'detected_as_expected': sum(row.get('before') == expected for row in group),
                    'repaired': sum(row.get('repaired', False) for row in group),
                    'blocked': sum(row['complete'] and not row['allowed'] for row in group),
                    'judge_after_pass': sum(row.get('judge_after') == 'pass' for row in group)}, ensure_ascii=False))
    return {'scope': manifest['scope'], 'results': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'evaluate', 'status', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-dir', type=Path, default=SOURCE)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--run', choices=RUNS)
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
    if not args.execute_api:
        print('Read-only; add --execute-api')
        return
    evaluate = args.action == 'evaluate'
    model, url = ('Qwen3.8-27B', os.environ.get('DRAMA_EVAL_BASE_URL', 'http://127.0.0.1:8001/v1')) if evaluate else ('Qwen3.6-27B', os.environ.get('DRAMA_LLM_BASE_URL', 'http://127.0.0.1:8000/v1'))
    base.verify_server(url, model)
    tasks = [(case, run) for case in cases for run in RUNS if not args.run or run == args.run]
    if evaluate:
        for case, run in tasks:
            read_outcome(make_gate(root, case, run))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(asyncio.run, run_case(root, case, run, evaluate)): (case['id'], run) for case, run in tasks}
        for future in as_completed(futures):
            try:
                print('completed=' + future.result(), flush=True)
            except Exception as error:
                failures.append(futures[future])
                print(json.dumps({'task': futures[future], 'error': str(error)}, ensure_ascii=False), flush=True)
    report(root)
    if failures:
        raise SystemExit('Some requests failed; valid stages retained; rerun the same command')


if __name__ == '__main__':
    main()
