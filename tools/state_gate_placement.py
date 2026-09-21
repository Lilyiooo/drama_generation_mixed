"""E45 known-error detection and single-repair placement diagnostic."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = Path(os.environ.get('SCRIPTPIPELINE_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
from tools import state_constraint_branch as base
from service.drama_by_creativity.scene_outline_inference import validate_repair_preserves_scenes

ROOT = REPO / 'output/state_gate_placement_e45_v2'
SOURCE = REPO / 'output/state_constraint_branch_e45_v1'
RUNS = base.RUNS
ARMS = ('raw_scene', 'scene_gate', 'script_gate')
SCENE_ERROR_QUOTE = '白无相突然从人群中跳上高台'
CHECKS = ('error_scene', 'error_script', 'clean_scene', 'clean_script')


def dependencies():
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(base.__file__), REPO / 'service/drama_by_creativity/scene_outline_inference.py')}


def load(root):
    manifest, inputs = base.read(root / 'manifest.json'), base.read(root / 'inputs.json')
    if manifest['inputs_sha256'] != base.digest(inputs):
        raise ValueError('Frozen inputs changed')
    if manifest['implementation_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or manifest['dependencies'] != dependencies():
        raise ValueError('Experiment implementation changed; use a fresh output directory')
    return manifest, inputs


def prepare(source, root):
    if (root / 'manifest.json').exists():
        load(root)
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError('Refusing to overwrite a nonempty unregistered directory')
    parent_manifest, frozen = base.load(source)
    original_root = Path(parent_manifest['source'])
    for relative, checksum in parent_manifest['source_files_sha256'].items():
        if hashlib.sha256((original_root / relative).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Original source changed since previous experiment: {relative}')
    paths = list(original_root.glob(f"diagnostics/*/model_calls/{base.TRACE_IDS['scene']}/output.txt"))
    if len(paths) != 1:
        raise ValueError('Expected exactly one original scene response')
    raw_scene = paths[0].read_text(encoding='utf-8')
    scenes = base.validate_scene_outline(raw_scene, 45)
    if SCENE_ERROR_QUOTE not in raw_scene:
        raise ValueError('Known-error scene evidence changed')
    script_records = list(original_root.glob(f"diagnostics/*/model_calls/{base.TRACE_IDS['script']}/record.json"))
    if len(script_records) != 1:
        raise ValueError('Expected exactly one original script request')
    original_script_request = base.read(script_records[0])
    original_scene_input = original_script_request['metadata']['parameters']['AllSceneOutlines']
    if json.loads(original_scene_input) != scenes:
        raise ValueError('Original script did not use the frozen erroneous scenes')
    replay = base.generation_payload(frozen, 'script', scenes, False, 4601)
    if replay['messages'] != original_script_request['request']['messages']:
        raise ValueError('Raw-scene control must exactly reproduce the original script prompts')
    files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        source / 'inputs.json', source / 'manifest.json', paths[0], script_records[0])}
    clean = {}
    for run in RUNS:
        scene_path = source / 'generation/control' / run / 'scene.json'
        script_path = source / 'variants/control' / f'{run}.json'
        scene_record = base.read(scene_path)
        scene_value = base.validate_scene_outline(scene_record['raw'], 45)
        if scene_value != scene_record['result']:
            raise ValueError('Clean scene cache changed')
        script = base.candidate(source, {'arm': 'control', 'run': run}, frozen)
        clean[run] = {'scene': scene_value, 'script': script}
        for path in (scene_path, script_path):
            files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = {'context': frozen, 'original_scenes': scenes, 'clean_controls': clean}
    candidates = [(arm, run, 'script') for run in RUNS for arm in ARMS]
    candidates += [('scene_gate', run, 'scene') for run in RUNS]
    candidates += [('original', 'reference', kind) for kind in ('scene', 'script')]
    random.Random(4618).shuffle(candidates)
    manifest = {
        'version': 1, 'source': str(source), 'inputs_sha256': base.digest(inputs), 'source_files_sha256': files,
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'dependencies': dependencies(),
        'generator': 'Qwen3.6-27B', 'evaluator': 'Qwen3.8-27B',
        'seeds': {'R01': 4601, 'R02': 4602, 'R03': 4603},
        'blind_map': {f'B{index + 1:02d}': {'arm': arm, 'run': run, 'kind': kind}
                      for index, (arm, run, kind) in enumerate(candidates)},
        'scope': 'Known-error E45 local diagnostic; repeated checks of the same error are not independent cases',
        'ground_truth': {'error_scene': 'fail', 'error_script': 'fail', 'clean_scene': 'pass', 'clean_script': 'pass'},
        'control_note': 'Clean scenes manually inspected for C01; clean scripts passed previous Qwen3.8 review. Not proof of no other errors.',
        'policy': 'At most one scene or script rewrite, only on fail/uncertain. Preserve misses and failed repairs.',
    }
    base.atomic_json(root / 'inputs.json', inputs)
    base.atomic_json(root / 'manifest.json', manifest)


def scene_text(scenes):
    return json.dumps(scenes, ensure_ascii=False, indent=2)


def check_payload(context, text, kind, model, seed=None):
    payload = base.judge_payload(context, text, model, evaluation=model == 'Qwen3.8-27B' and kind == 'script')
    if kind == 'scene':
        payload['messages'][0]['content'] = base.CHECK_RULES.replace('候选正文', '候选场次大纲').replace('候选文本', '候选场次大纲')
        payload['messages'][0]['content'] += '\n当前候选是场次大纲：检查它计划安排的当前现实行动，不要把明确标注的回忆或遗留资料当作活人出场。'
        packet = json.loads(payload['messages'][1]['content'])
        packet['candidate_scene_outline'] = packet.pop('candidate_script')
        payload['messages'][1]['content'] = json.dumps(packet, ensure_ascii=False)
    if seed is not None:
        payload['seed'] = seed
    payload['messages'][0]['content'] += (
        '\n证据格式要求：evidence_quote 只复制候选中的一段短小、连续原文，优先选择直接体现行动或对白的片段。'
        '不要把 JSON 字段名和数组重新排版为引文，不要拼接不同位置、改写标点或添加省略号。'
        '每一条 violations 都会独立校验；不要额外列入未逐字核实的引文。'
        '仅出现在人物名单中不能单独证明现实行动；不得只引用名单、在 reason 中补写行动作为证据。'
        '若返回格式错误，请按反馈重新核对候选，不得为了通过格式校验而改判 pass 或删除确有依据的冲突。')
    return payload


def parse_check(raw, text, kind, model):
    # Give the schema retry actionable feedback without supplying a verdict or evidence.
    value = json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
    violations = value.get('violations') if isinstance(value, dict) else None
    if isinstance(violations, list):
        errors = []
        for index, item in enumerate(violations):
            if not isinstance(item, dict):
                continue
            quote = item.get('evidence_quote')
            reason = item.get('reason')
            if not isinstance(quote, str) or not quote.strip() or quote not in text:
                errors.append(f'violations[{index}].evidence_quote is not an exact quote '
                              f'from the candidate: {json.dumps(quote, ensure_ascii=False)[:240]}')
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f'violations[{index}].reason must be a nonempty string')
        if errors:
            raise ValueError('; '.join(errors) +
                             '。请从候选内重新复制连续原文；不要重排 JSON 字段、拼接或改写引文。'
                             '保持基于证据的判断，不要为了格式通过而改判 pass。')
    return base.validate_check(raw, text, evaluation=model == 'Qwen3.8-27B' and kind == 'script')


def examples(inputs, run):
    return {
        'error_scene': ('scene', scene_text(inputs['original_scenes'])),
        'error_script': ('script', inputs['context']['original_script']),
        'clean_scene': ('scene', scene_text(inputs['clean_controls'][run]['scene'])),
        'clean_script': ('script', inputs['clean_controls'][run]['script']),
    }


def checked_request(directory, name, context, text, kind, model, seed, url):
    return base.request_cached(directory, name, check_payload(context, text, kind, model, seed), url,
                               lambda raw: parse_check(raw, text, kind, model), base.digest(text))


def repair_payload(context, scenes, check, kind, seed):
    if kind == 'scene':
        payload = base.generation_payload(context, 'scene', None, False, seed)
        draft = scene_text(scenes)
        instruction = ('修订完整场次大纲JSON，保留原场次数量、编号顺序和字段结构。不要重新抽取一套无关情节。'
                       '纠正当前时间的角色行动安排，可将必要行动交由在世角色承担。')
    else:
        payload = base.generation_payload(context, 'script', scenes, False, seed)
        draft = context['original_script']
        instruction = ('修订完整第45集正文，仅输出正文。允许纠正上游错误场次中安排的行动，不要继续照抄错误安排。')
    payload['messages'] += [
        {'role': 'assistant', 'content': draft},
        {'role': 'user', 'content': instruction + '\n检查结果：' + json.dumps(check, ensure_ascii=False)
         + base.constraint_block(context)
         + '\n保留军饷证据、说服禁军倒戈、萧砚接管指挥权等本集核心任务以及原字数要求。不要删光冲突来规避问题。'},
    ]
    return payload


def validate_scene_repair(raw, original):
    scenes = base.validate_scene_outline(raw, 45)
    validate_repair_preserves_scenes(scene_text(original), scenes)
    return scenes


def write_result(path, value):
    if path.exists() and base.read(path) != value:
        raise ValueError(f'Refusing to overwrite a different result: {path}')
    base.atomic_json(path, value)


def generate_run(root, run, url):
    manifest, inputs = load(root)
    context, original = inputs['context'], inputs['original_scenes']
    seed = manifest['seeds'][run]
    directory = root / 'generation' / run
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checks = {}
        for name, (kind, text) in examples(inputs, run).items():
            checks[name] = checked_request(directory, name, context, text, kind, 'Qwen3.6-27B', seed, url)
        raw_script = base.request_cached(directory, 'raw_scene_script',
            base.generation_payload(context, 'script', original, False, seed), url, base.plain_text, base.digest(original))
        metadata = {'inputs_sha256': manifest['inputs_sha256'], 'seed': seed}
        base.write_variant(root / 'variants/raw_scene' / f'{run}.json', raw_script, metadata)

        scene_check = checks['error_scene']
        scene_rewritten = scene_check['status'] != 'pass'
        scenes = original
        if scene_rewritten:
            scenes = base.request_cached(directory, 'scene_repair',
                repair_payload(context, original, scene_check, 'scene', seed), url,
                lambda raw: validate_scene_repair(raw, original), base.digest(scene_check))
            scene_check = checked_request(directory, 'scene_after', context, scene_text(scenes), 'scene', 'Qwen3.6-27B', seed, url)
        scene_script = raw_script if scenes == original else base.request_cached(directory, 'scene_gate_script',
            base.generation_payload(context, 'script', scenes, False, seed), url, base.plain_text, base.digest(scenes))
        scene_metadata = {**metadata, 'repair_attempted': scene_rewritten, 'text_changed': scenes != original, 'final_check': scene_check}
        write_result(root / 'scenes/scene_gate' / f'{run}.json', {'scenes': scenes, 'scene_sha256': base.digest(scenes), **scene_metadata})
        base.write_variant(root / 'variants/scene_gate' / f'{run}.json', scene_script, scene_metadata)

        script_check = checks['error_script']
        script_rewritten = script_check['status'] != 'pass'
        final = context['original_script']
        if script_rewritten:
            final = base.request_cached(directory, 'script_repair',
                repair_payload(context, original, script_check, 'script', seed), url, base.plain_text, base.digest(script_check))
            script_check = checked_request(directory, 'script_after', context, final, 'script', 'Qwen3.6-27B', seed, url)
        base.write_variant(root / 'variants/script_gate' / f'{run}.json', final, {
            **metadata, 'repair_attempted': script_rewritten, 'text_changed': final != context['original_script'], 'final_check': script_check})
    return run


def candidate(root, mapping, inputs):
    kind, arm, run = mapping['kind'], mapping['arm'], mapping['run']
    if arm == 'original':
        return scene_text(inputs['original_scenes']) if kind == 'scene' else inputs['context']['original_script']
    if kind == 'scene':
        record = base.read(root / 'scenes' / arm / f'{run}.json')
        if record['scene_sha256'] != base.digest(record['scenes']):
            raise ValueError('Scene candidate changed')
        result = scene_text(record['scenes'])
    else:
        record = base.read(root / 'variants' / arm / f'{run}.json')
        if record['script_sha256'] != base.digest(record['script']):
            raise ValueError('Script candidate changed')
        result = record['script']
    if record['inputs_sha256'] != base.digest(inputs):
        raise ValueError('Candidate belongs to different inputs')
    return result


def evaluate_one(root, blind_id, url):
    manifest, inputs = load(root)
    mapping = manifest['blind_map'][blind_id]
    text = candidate(root, mapping, inputs)
    directory = root / 'evaluation' / blind_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked_request(directory, 'judge', inputs['context'], text, mapping['kind'], 'Qwen3.8-27B', None, url)
    return blind_id


def verified_check(path, context, text, kind, model, seed):
    record = base.read(path)
    expected = base.digest({'payload': check_payload(context, text, kind, model, seed),
                            'dependency': base.digest(text), 'base_url': record['base_url']})
    if record['fingerprint'] != expected or record['dependency'] != base.digest(text):
        raise ValueError(f'Judge inputs changed: {path}')
    result = parse_check(record['raw'], text, kind, model)
    if record['result'] != result:
        raise ValueError(f'Judge result changed: {path}')
    return result


def report(root):
    manifest, inputs = load(root)
    rows, detections = [], []
    for run in RUNS:
        for name, (kind, text) in examples(inputs, run).items():
            path = root / 'generation' / run / f'{name}.json'
            row = {'run': run, 'case': name, 'expected': manifest['ground_truth'][name], 'complete': path.exists()}
            if path.exists():
                row.update(verified_check(path, inputs['context'], text, kind, 'Qwen3.6-27B', manifest['seeds'][run]))
            detections.append(row)
    for blind_id, mapping in manifest['blind_map'].items():
        row = {**mapping, 'generated': False, 'scored': False}
        try:
            text = candidate(root, mapping, inputs)
        except FileNotFoundError:
            rows.append(row)
            continue
        row.update(generated=True, characters=len(text))
        if mapping['arm'] != 'original':
            folder = 'scenes' if mapping['kind'] == 'scene' else 'variants'
            record = base.read(root / folder / mapping['arm'] / f"{mapping['run']}.json")
            row.update(repair_attempted=record.get('repair_attempted', False), text_changed=record.get('text_changed'))
            if 'final_check' in record:
                row.update(gate_check_status=record['final_check']['status'],
                           gate_check_kind='scene' if mapping['arm'] == 'scene_gate' else 'script')
        path = root / 'evaluation' / blind_id / 'judge.json'
        if path.exists():
            row.update(verified_check(path, inputs['context'], text, mapping['kind'], 'Qwen3.8-27B', None), scored=True)
        rows.append(row)
    for name in CHECKS:
        group = [row for row in detections if row['case'] == name]
        print(json.dumps({'detector_case': name, 'expected': manifest['ground_truth'][name],
            'complete': sum(row['complete'] for row in group), 'total': len(group),
            'pass': sum(row.get('status') == 'pass' for row in group),
            'fail': sum(row.get('status') == 'fail' for row in group),
            'uncertain': sum(row.get('status') == 'uncertain' for row in group)}, ensure_ascii=False))
    for kind, arms in [('script', (*ARMS, 'original')), ('scene', ('scene_gate', 'original'))]:
        for arm in arms:
            group = [row for row in rows if row['kind'] == kind and row['arm'] == arm]
            scored = [row for row in group if row['scored']]
            print(json.dumps({'kind': kind, 'arm': arm, 'generated': sum(row['generated'] for row in group),
                'total': len(group), 'scored': len(scored),
                'repairs': sum(row.get('repair_attempted', False) for row in group),
                'pass': sum(row.get('status') == 'pass' for row in group),
                'fail': sum(row.get('status') == 'fail' for row in group),
                'uncertain': sum(row.get('status') == 'uncertain' for row in group),
                'outline_mean': sum(row['outline_alignment'] for row in scored) / len(scored) if scored and kind == 'script' else None,
                'quality_mean': sum(row['writing_quality'] for row in scored) / len(scored) if scored and kind == 'script' else None}, ensure_ascii=False))
    return {'scope': manifest['scope'], 'detections': detections, 'results': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'generate', 'evaluate', 'status', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-dir', type=Path, default=SOURCE)
    parser.add_argument('--run', choices=RUNS)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--execute-api', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('--workers must be between 1 and 4')
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(args.source_dir.resolve(), root)
        print(f'Prepared {root}; source results unchanged; no API calls')
        return
    manifest, inputs = load(root)
    result = report(root)
    if args.action in {'status', 'report'}:
        if args.action == 'report':
            base.atomic_json(root / 'summary.json', result)
        return
    if not args.execute_api:
        print('Read-only; add --execute-api for model calls')
        return
    if args.action == 'generate':
        url = os.environ.get('DRAMA_LLM_BASE_URL', 'http://127.0.0.1:8000/v1')
        base.verify_server(url, 'Qwen3.6-27B')
        tasks = [run for run in RUNS if not args.run or run == args.run]
        operation = lambda task: generate_run(root, task, url)
    else:
        url = os.environ.get('DRAMA_EVAL_BASE_URL', 'http://127.0.0.1:8001/v1')
        base.verify_server(url, 'Qwen3.8-27B')
        tasks = [key for key, mapping in manifest['blind_map'].items() if not args.run or mapping['run'] in {args.run, 'reference'}]
        for task in tasks:
            candidate(root, manifest['blind_map'][task], inputs)
        operation = lambda task: evaluate_one(root, task, url)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(operation, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                print('completed=' + future.result(), flush=True)
            except Exception as error:
                failures.append(futures[future])
                print(json.dumps({'task': futures[future], 'error': str(error)}, ensure_ascii=False), flush=True)
    report(root)
    if failures:
        raise SystemExit('Some tasks failed; successful stages retained; rerun the same command')


if __name__ == '__main__':
    main()
