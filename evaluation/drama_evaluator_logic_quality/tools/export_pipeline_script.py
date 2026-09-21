"""Export exactly 60 episode JSON files without modifying their script contents."""
import argparse
import hashlib
import json
from pathlib import Path


def export(source, target, expected=60):
    folder = source/'05_drama'
    files = list(folder.glob('episode_*.json'))
    expected_names = {f'episode_{i:02d}.json' for i in range(1, expected+1)}
    if {f.name for f in files} != expected_names:
        raise ValueError(f'需要完整{expected}集，实际文件{len(files)}个；缺少：{sorted(expected_names-{f.name for f in files})}')
    texts, provenance = [], []
    for i in range(1, expected+1):
        path = folder/f'episode_{i:02d}.json'
        raw = path.read_bytes()
        row = json.loads(raw)
        text = row.get('content')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'第{i}集正文为空')
        if row.get('episode_id') != i-1:
            raise ValueError(f'第{i}集内部episode_id与文件名不匹配')
        texts.append(f'第{i}集\n\n'+text)
        provenance.append(dict(episode=i, path=str(path), sha256=hashlib.sha256(raw).hexdigest(), characters=len(text)))
    script = '\n\n'.join(texts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding='utf-8')
    target.with_suffix('.source.json').write_text(json.dumps(dict(source=str(source), episodes=provenance,
        script_sha256=hashlib.sha256(script.encode()).hexdigest(), characters=len(script)),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'已导出{expected}集，共{len(script)}字符：{target}', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('target',type=Path)
    args=p.parse_args()
    export(args.source.resolve(),args.target.resolve())
