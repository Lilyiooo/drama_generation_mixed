"""保留 01—03，备份后续产物后在原目录重跑阶段规划、集大纲和正文。"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from drama_local import models as pb
from drama_local.runtime import Context, atomic_json
from run_from_episode_outline import build_story_outline, build_generate_input
from run_test import create_creativity, get_plot_type


def load_inputs(source):
    story = build_story_outline(str(source))
    # 旧输出可恢复背景参数；新流程不需要样稿。
    for path in sorted(source.glob('diagnostics/*/model_calls/*/record.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        params = record.get('metadata', {}).get('parameters', {})
        if 'StartEpisodeId' in params:
            return story, params
    return story, {}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--episode-nums', type=int, default=60)
    parser.add_argument('--story-id', default='APID-test-001')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    source = Path(args.source_dir).resolve(strict=True)
    story, params = load_inputs(source)
    from service.drama_by_creativity.stage_episode_planning import extract_stages
    stages = extract_stages(story.story_outline[0])
    if args.episode_nums < len(stages):
        raise ValueError('总集数不能小于阶段数')
    generate_input = build_generate_input(args.episode_nums)
    generate_input.core_story = params.get('StoryOutline') or generate_input.core_story
    generate_input.topic = params.get('Topic') or generate_input.topic
    generate_input.world_view = params.get('WorldView', '')
    print(f'[输入] {source}/03_story_outline/outline.json', flush=True)
    print(f'[阶段] {stages.keys() if isinstance(stages, dict) else [s["stage"] for s in stages]}', flush=True)
    if args.check_only:
        print('输入检查通过，未修改输出文件、未调用模型。')
        return

    backup = source.with_name(source.name + '_before_stage_rerun_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir()
    keep = {'01_script_proposal', '02_demo_drama', '03_story_outline'}
    moved = []
    try:
        for entry in source.iterdir():
            if entry.name not in keep:
                entry.rename(backup / entry.name)
                moved.append(entry.name)
    except BaseException:
        for name in reversed(moved):
            (backup / name).rename(source / name)
        raise
    os.environ['DRAMA_OUTPUT_DIR'] = str(source)
    atomic_json(source / 'resume_from_story_outline.json', {
        'source_dir': str(source), 'backup_dir': str(backup),
        'episode_nums': args.episode_nums,
        'summary_memory_baseline': os.environ.get('DRAMA_SUMMARY_MEMORY_BASELINE', 'false'),
    })
    print(f'[旧后续内容备份] {backup}', flush=True)
    print(f'[新结果输出] {source}', flush=True)
    started = time.time()
    status = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'success': False}
    try:
        creativity = create_creativity()
        ctx = Context()
        info = pb.StoryInfo(story_id=args.story_id, plot_type=get_plot_type('SHORT_CARTOON'))
        outline = await creativity.GenerateEpisodeOutline(ctx, pb.GenerateEpisodeOutlineByCreativityReq(
            story_info=info, generate_input=generate_input,
            generated_data=pb.GeneratedData(story_outline=story)))
        if not outline or not outline.episode_outline.seasons:
            raise RuntimeError('集大纲生成失败，请查看 diagnostics')
        episodes = outline.episode_outline.seasons[0].episodes
        if [ep.episode_id for ep in episodes] != list(range(args.episode_nums)):
            raise RuntimeError('集大纲集号不完整，停止正文生成')
        result = await creativity.GenerateDrama(ctx, pb.GenerateDramaByCreativityReq(
            story_info=info, generate_input=generate_input,
            generate_data=pb.GenerateDrama(story_outline=story, episode_outline=outline.episode_outline,
                prev_episode_content='', suggestion='', select_range=[
                    pb.GenerateDrama.SelectRange(season_id=0, episode_ids=list(range(args.episode_nums)))])))
        generated = result.result.seasons[0].episodes if result and result.result and result.result.seasons else []
        actual = sorted(ep.episode_id for ep in generated)
        status['generated_episode_ids'] = actual
        if actual != list(range(args.episode_nums)):
            raise RuntimeError(f'正文未完整生成 {args.episode_nums} 集，实际集号（0-based）：{actual}')
        status['success'] = True
    except BaseException as error:
        status['error'] = str(error)
        raise
    finally:
        status['elapsed_seconds'] = round(time.time() - started, 2)
        status['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(source / 'generation_status.json', status)
        print(f'[生成状态] {status}', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
