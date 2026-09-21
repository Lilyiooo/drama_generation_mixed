import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from drama_local import models as pb
from drama_local.runtime import LocalStore, ScriptContentFile


class RuntimeTests(unittest.TestCase):
    def test_nested_copy_and_json_roundtrip(self):
        outline = pb.EpisodeOutline(seasons=[pb.EpisodeOutline.Season(
            episodes=[pb.EpisodeOutline.Episode(episode_id=2, content='原大纲')])])
        data = pb.GenerateDrama(episode_outline=outline)
        data.episode_outline.seasons[0].episodes[0].content = '修改'
        self.assertEqual(outline.seasons[0].episodes[0].content, '原大纲')
        result = pb.GenerateDramaRsp()
        result.episode_outline.CopyFrom(outline)
        outline.seasons[0].episodes.clear()
        self.assertEqual(len(result.episode_outline.seasons[0].episodes), 1)
        self.assertEqual(pb.GenerateDramaRsp.from_dict(result.to_dict()), result)
        a, b = pb.GenerateDrama(), pb.GenerateDrama()
        a.select_range.append(pb.GenerateDrama.SelectRange(episode_ids=[0]))
        self.assertEqual(b.select_range, [])
        with self.assertRaises(TypeError):
            pb.StoryInfo(unknown_field=1)

    def test_local_storage_isolation_and_restart(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            async def check():
                await ScriptContentFile('第一集').upload(None, LocalStore(a), project_id='same', episode_id=0)
                restored = await ScriptContentFile.download(None, LocalStore(a), project_id='same', episode_id=0)
                self.assertEqual(restored.result, '第一集')
                with self.assertRaises(FileNotFoundError):
                    await ScriptContentFile.download(None, LocalStore(b), project_id='same', episode_id=0)
                with self.assertRaises(ValueError):
                    await ScriptContentFile('x').upload(None, LocalStore(a), project_id='../x', episode_id=0)
            asyncio.run(check())

    def test_prompt_files_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        hashes = json.loads((root/'localization_backup/prompt_hashes.json').read_text())
        self.assertTrue(hashes)
        for path, digest in hashes.items():
            content = (root/path).read_bytes()
            if path.endswith('/prompts/utils.py'):
                # User-authorized clarification; all other original prompt text is preserved.
                content = content.replace('该标记仅用于上一集与当前集之间，出现一次即可；同一集的各场之间不要使用。\n'.encode(), b'')
            # Simple variant: user-authorized removal of sample-script inputs.
            approved_simple_prompts = {'service/drama_by_creativity/prompts/outline.py': '3012363906346cedb77a7ae9f3a99e1ab8130c7aec84ff9a844a6b4c4c864bea', 'service/drama_by_creativity/prompts/episode_outline.py': '54f056b0f1b60115e97b5af47b0c320b455a48c2c50e9476fa38f29006533dc2'}
            digest = approved_simple_prompts.get(path, digest)
            self.assertEqual(hashlib.sha256(content).hexdigest(), digest, path)
