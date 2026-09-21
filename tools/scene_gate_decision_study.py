"""Paired, check-only calibration of indexed and factorized state decisions."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from drama_local.runtime import Context, atomic_json
from service.drama_by_creativity import scene_gate_decision as decision
from service.drama_by_creativity import scene_state_gate as gate
from tools import repair_scene_gate_calibration as indexed
from tools import scene_gate_calibration as original

ROOT = REPO / 'output/scene_gate_decision_v2'
SOURCE = REPO / 'output/scene_gate_multitype_v1'
ARMS = ('control_indexed', 'candidate_factorized')
RUNS = ('R01', 'R02', 'R03')


def dependencies():
    return {**indexed.dependencies(), 'decision': decision.implementation_hash(),
            'study': indexed.file_hash(Path(__file__))}


def make_scene(action, location, cast):
    return {'场次编号': '2-1', '时间与地点': location, '核心功能': '推动当前调查',
            '情节概要': action, '主要情节': [action], '悬念钩子': '调查将发现什么？',
            '出场人物': cast, '冲突与张力': ['调查时间紧迫。'],
            '人物动机与目标': [{'角色': name, '动机与目标': '推进调查'} for name in cast]}


def controlled_cases():
    specifications = [
        {'domain': 'life', 'field': 'character_state',
         'fact': '本集开始时，李舟已经死亡并安葬。',
         'opposite': '本集开始时，李舟活着，能够正常行动。',
         'missing': '本集开始时，李舟此前失联，生死和伤势尚未确认。',
         'bad': '李舟本人在当前现实时间推开门，与周岚面对面说话，随后一起离开；不是回忆、幻象或他人扮演。',
         'good': '周岚独自翻阅李舟生前留下的笔记，读出其中线索，然后独自离开；李舟本人没有出现。',
         'location': '书房门外 日', 'bad_cast': ['李舟', '周岚'], 'good_cast': ['周岚']},
        {'domain': 'object', 'field': 'resources_and_evidence',
         'fact': '本集开始时，唯一的原始账本已经烧成灰烬且无法恢复，周岚持有此前抄录的副本。',
         'opposite': '本集开始时，同一本唯一的原始账本完好无损，仍由周岚持有。',
         'missing': '本集开始时，周岚持有账本文稿，但尚未确认它是原始账本还是副本，也未确认原件是否损毁。',
         'bad': '周岚直接取出那本唯一的原始账本，翻开完好的纸页供众人核对，明确不是副本，也没有修复过程。',
         'good': '周岚说明原始账本已毁，取出此前抄录的副本供李舟核对，不使用原件。',
         'location': '档案室 日', 'bad_cast': ['周岚', '李舟'], 'good_cast': ['周岚', '李舟']},
        {'domain': 'resource', 'field': 'resources_and_evidence',
         'fact': '本集开始时，周岚钱袋现金为零，没有其他现金或代付渠道；她拥有一支可典当的玉簪。',
         'opposite': '本集开始时，同一个钱袋中已有十两银子可直接使用。',
         'missing': '本集开始时，周岚携带的钱袋余额尚未核实，可能为空，也可能装有银子。',
         'bad': '周岚没有典当、借贷、他人代付或获得新收入，直接从该钱袋取出十两银子付给守卫。',
         'good': '周岚在当铺柜台将自己拥有的玉簪交给铺主。铺主验货，出具当票并当场付给她十两银子。周岚接过这笔新钱，付给等在柜台旁的守卫。',
         'location': '当铺柜台 日', 'bad_cast': ['周岚', '守卫'], 'good_cast': ['周岚', '铺主', '守卫']},
        {'domain': 'knowledge', 'field': 'unknown_information',
         'fact': '本集开始时，周岚不知道密室口令，尚未取得记载口令的资料。',
         'opposite': '本集开始时，周岚已经准确知道同一间密室的口令。',
         'missing': '本集开始时，尚未确认周岚是否知道密室口令，也未确认她是否拥有口令资料。',
         'bad': '周岚没有获得提示或资料，也没有猜测推理，直接凭自己早已掌握的准确口令打开密室门。',
         'good': '在密室门外，守卫当场将写有正确口令的纸条交给周岚。周岚第一次读到口令，随后使用刚获得的口令打开门，进入密室。',
         'location': '密室门外 日', 'bad_cast': ['周岚'], 'good_cast': ['周岚', '守卫']},
    ]
    cases = []
    for spec in specifications:
        for variant, expected, kind in [('hard', 'fail', 'hard_conflict'), ('legal', 'pass', None),
                                        ('memory', 'uncertain', 'memory_conflict'), ('missing', 'uncertain', 'insufficient_evidence')]:
            view = {field: [] for field in gate.FIELDS}
            view[spec['field']] = [spec['missing'] if variant == 'missing' else spec['fact']]
            if variant == 'memory':
                view[spec['field']].append(spec['opposite'])
            action = spec['good'] if variant == 'legal' else spec['bad']
            cast = spec['good_cast'] if variant == 'legal' else spec['bad_cast']
            if variant == 'legal':
                kind = 'legal_transition' if spec['domain'] in {'resource', 'knowledge'} else 'no_conflict'
            cases.append({'id': spec['domain'] + '_' + variant, 'origin': 'controlled_synthetic_v2',
                          'domain': spec['domain'], 'expected_status': expected, 'expected_kind': kind,
                          'episode': 2, 'view': view, 'scenes': [make_scene(action, spec['location'], cast)],
                          'params': {'Outline': '角色通过当前行动推进调查。', 'PrevEpisodeScript': '', 'EpisodeIdx': '2',
                                     'WorldView': '架空古代调查故事', 'RoleInfo': '本场角色以场次记载为准。'}})
    return cases


def prepare(source, root):
    if (root / 'manifest.json').exists():
        load(root)
        return
    if root == source or (root.exists() and any(root.iterdir())):
        raise ValueError('Use a fresh study directory')
    _, previous_cases = original.load(source)
    cases = controlled_cases()
    real_cases = [case for case in previous_cases if case['origin'] == 'real_saved_generation']
    if len(real_cases) != 2:
        raise ValueError('Expected two real source scenes')
    for case in real_cases:
        cases.append({key: value for key, value in case.items() if key not in {'expected', 'category'}} |
                     {'domain': case['category'], 'expected_status': None, 'expected_kind': None,
                      'label_note': 'Prior label concerned death-action only, not all retrieved state; diagnostic, not overall gold.'})
    for case in cases:
        gate.state_catalog(case['view'])
        gate.validate_scene_outline(json.dumps(case['scenes'], ensure_ascii=False), case['episode'])
    files = [source / 'manifest.json', source / 'cases.json']
    atomic_json(root / 'cases.json', cases)
    atomic_json(root / 'manifest.json', {'source_files': {str(path): indexed.file_hash(path) for path in files},
                'cases_sha256': gate.digest(cases), 'implementation': dependencies(), 'arms': list(ARMS), 'runs': list(RUNS),
                'scope': '16 revised controlled cases + 2 unchanged real diagnostic scenes; 3 repeats; check only, no repair or whole-series efficacy claim.'})


def load(root):
    manifest, cases = gate.read(root / 'manifest.json'), gate.read(root / 'cases.json')
    if manifest['implementation'] != dependencies() or manifest['cases_sha256'] != gate.digest(cases):
        raise ValueError('Frozen study implementation or cases changed; use a fresh directory')
    for name, expected in manifest['source_files'].items():
        if indexed.file_hash(Path(name)) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    return manifest, cases


class DecisionProbe(gate.SceneStateGate):
    def __init__(self, root, case, run, arm):
        super().__init__(root / arm / case['id'] / run, case['episode'], case['params'], case['view'], 'audit')
        self.binding['study_implementation'] = dependencies()
        self.arm = arm
        self.model = 'Qwen3.8-27B' if arm == 'judge' else 'Qwen3.6-27B'
        self.endpoint = os.environ.get('DRAMA_EVAL_BASE_URL', 'http://127.0.0.1:8001/v1') if arm == 'judge' else self.base_url
        self.scenes = case['scenes']

    def packet(self, scenes):
        return {**super().packet(scenes), 'scene_evidence_catalog': indexed.evidence_catalog(scenes)}

    def protocol(self):
        if self.arm == 'control_indexed':
            return indexed.RULES, lambda raw: indexed.validate_indexed(raw, self.scenes, self.catalog)
        return decision.RULES, lambda raw: decision.validate_decision(raw, self.scenes, self.catalog)

    async def run(self):
        system, validator = self.protocol()
        return await self.request(Context(), 'decision', system, json.dumps(self.packet(self.scenes), ensure_ascii=False),
                                  validator, self.model, self.endpoint)

    def read_result(self):
        cached = gate.read(self.directory / 'decision.json')
        request = cached['request']
        system, validator = self.protocol()
        if (cached['fingerprint'] != gate.digest(request) or request['binding'] != self.binding or
                request['model'] != self.model or request['system'] != system or
                request['user'] != json.dumps(self.packet(self.scenes), ensure_ascii=False)):
            raise ValueError('Cached decision input or model changed')
        checked = validator(cached['raw'])
        if checked != cached['result']:
            raise ValueError('Cached decision result changed')
        return checked


def report(root):
    manifest, cases = load(root)
    rows = []
    for case in cases:
        for run in RUNS:
            judge = DecisionProbe(root, case, run, 'judge')
            judged = judge.read_result() if (judge.directory / 'decision.json').exists() else None
            for arm in ARMS:
                probe = DecisionProbe(root, case, run, arm)
                row = {'case': case['id'], 'run': run, 'arm': arm, 'origin': case['origin'],
                       'expected_status': case['expected_status'], 'expected_kind': case['expected_kind'], 'complete': False}
                if (probe.directory / 'decision.json').exists():
                    result = probe.read_result()
                    row.update(complete=True, status=result['status'],
                               classifications=sorted({check['classification'] for check in result.get('checks', [])}))
                if judged is not None:
                    row['judge_status'] = judged['status']
                rows.append(row)
    for arm in ARMS:
        group = [row for row in rows if row['arm'] == arm]
        print(f"{arm}: complete={sum(row['complete'] for row in group)}/{len(group)}")
        for expected in ('fail', 'pass', 'uncertain'):
            labeled = [row for row in group if row['expected_status'] == expected]
            print(json.dumps({'arm': arm, 'expected_status': expected, 'total': len(labeled),
                              'complete': sum(row['complete'] for row in labeled),
                              'matched': sum(row.get('status') == expected for row in labeled)}, ensure_ascii=False))
        if arm == 'candidate_factorized':
            for kind in ('hard_conflict', 'legal_transition', 'no_conflict', 'memory_conflict', 'insufficient_evidence'):
                labeled = [row for row in group if row['expected_kind'] == kind]
                print(json.dumps({'expected_kind': kind, 'total': len(labeled),
                                  'complete': sum(row['complete'] for row in labeled),
                                  'kind_present': sum(kind in row.get('classifications', []) for row in labeled)}, ensure_ascii=False))
    judged_count = sum('judge_status' in row for row in rows if row['arm'] == ARMS[0])
    print(f'judge_complete={judged_count}/{len(cases) * len(RUNS)}; check_only=True auto_repairs=0')
    return {'scope': manifest['scope'], 'results': rows}


async def run_one(root, case, run, arm):
    probe = DecisionProbe(root, case, run, arm)
    probe.directory.mkdir(parents=True, exist_ok=True)
    with (probe.directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await probe.run()
    return f'{arm}/{case["id"]}/{run}'


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
    _, cases = load(root)
    summary = report(root)
    if args.action in {'status', 'report'}:
        if args.action == 'report':
            atomic_json(root / 'summary.json', summary)
        return
    evaluate = args.action == 'evaluate'
    if evaluate and any(not row['complete'] for row in summary['results'] if not args.run or row['run'] == args.run):
        raise ValueError('Complete both Qwen3.6 arms for the selected runs before judging')
    tasks = []
    for case in cases:
        for run in RUNS:
            if args.run and run != args.run:
                continue
            for arm in ('judge',) if evaluate else ARMS:
                probe = DecisionProbe(root, case, run, arm)
                if not (probe.directory / 'decision.json').exists():
                    tasks.append((case, run, arm))
    print(f'pending={len(tasks)} stage={args.action}')
    if not args.execute_api or not tasks:
        print('No API calls; add --execute-api if pending')
        return
    model, url = ('Qwen3.8-27B', os.environ.get('DRAMA_EVAL_BASE_URL', 'http://127.0.0.1:8001/v1')) if evaluate else ('Qwen3.6-27B', os.environ.get('DRAMA_LLM_BASE_URL', 'http://127.0.0.1:8000/v1'))
    original.base.verify_server(url, model)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(asyncio.run, run_one(root, case, run, arm)): (arm, case['id'], run)
                   for case, run, arm in tasks}
        for future in as_completed(futures):
            try:
                print('completed=' + future.result(), flush=True)
            except Exception as error:
                failures.append(futures[future])
                print(json.dumps({'task': futures[future], 'error': str(error)}, ensure_ascii=False), flush=True)
    report(root)
    if failures:
        raise SystemExit('Some checks failed; valid results retained; rerun the same command')


if __name__ == '__main__':
    main()
