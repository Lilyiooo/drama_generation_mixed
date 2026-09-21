"""Experimental factorized scene gate: one repair, semantic fallback, no API fallback."""

import hashlib
import json
from pathlib import Path

from drama_local.runtime import atomic_json
from . import scene_gate_decision as decision
from .scene_state_gate import SceneStateGate, digest
from .scene_outline_inference import validate_scene_outline, validate_repair_preserves_scenes
from tools.continue_scene_gate_decision import compatible_validate
from tools.repair_scene_gate_calibration import evidence_catalog


POLICY = 'factorized_one_repair_fallback_v1'
REPAIR_RULES = '''只修订 checks 中 classification=hard_conflict 的已证实冲突，其他写作建议和不确定项不得作为修改依据。
只输出完整场次JSON数组，保留原场次数量、编号、顺序、字段、核心任务、人物目标、冲突和字数规划。
禁止编造过去发生的复活、替身、资源补给、身份变化、提前知情或幕后事件来解释矛盾。
不得把未来大纲视为已经发生的事实；不得删空情节或改写世界观、人设和状态记忆。
可以由确实在场且具备能力和信息的角色承接必要行动，明确标注的回忆、遗物和合法场内转变可以保留。
这是唯一一轮语义修订；格式重试仍需修订同一份原始场次，不生成第二套改进方案。'''


class ConservativeSceneGate(SceneStateGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binding['experimental_policy'] = POLICY
        self.binding['experimental_implementation'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    @classmethod
    def factory(cls, memory, episode_index, params):
        return cls.from_memory(memory, episode_index, params, 'enforce')

    async def check(self, ctx, scenes, slot='check_before'):
        packet = {**self.packet(scenes), 'scene_evidence_catalog': evidence_catalog(scenes)}
        return await self.request(ctx, slot, decision.RULES, json.dumps(packet, ensure_ascii=False),
                                  lambda raw: compatible_validate(raw, scenes, self.catalog))

    async def apply(self, ctx, scenes):
        original = validate_scene_outline(json.dumps(scenes, ensure_ascii=False), self.episode)
        saved = self.saved_scenes()
        if saved is not None and saved != original:
            raise ValueError('Reuse the frozen original scenes when resuming a gate')
        if saved is None:
            atomic_json(self.directory / 'input.json', {
                'binding': self.binding, 'scenes': original, 'scenes_sha256': digest(original)})
        before = await self.check(ctx, original)
        after, selected, candidate = before, original, None
        if before['status'] == 'fail':
            hard_checks = [item for item in before['checks'] if item['classification'] == 'hard_conflict']
            if not hard_checks:
                raise ValueError('Repair requires a validated hard conflict')
            packet = {'original_generation_params': self.params, 'state_catalog': self.catalog,
                      'original_scenes': original, 'hard_checks': hard_checks}

            def validate_repair(raw):
                result = validate_scene_outline(raw, self.episode)
                validate_repair_preserves_scenes(json.dumps(original, ensure_ascii=False), result)
                return result

            candidate = await self.request(ctx, 'repair', REPAIR_RULES, json.dumps(packet, ensure_ascii=False),
                                           validate_repair, temperature=1)
            after = await self.check(ctx, candidate, 'check_after')
            if after['status'] == 'pass':
                selected = candidate
        outcome = {
            'policy': POLICY, 'binding_sha256': digest(self.binding),
            'original_scenes_sha256': digest(original), 'before': before, 'after': after,
            'repair_attempted': candidate is not None, 'repair_accepted': candidate is not None and after['status'] == 'pass',
            'fallback_to_original': candidate is not None and after['status'] != 'pass',
            'unresolved': after['status'] != 'pass', 'text_changed': selected != original,
            'candidate_scenes': candidate, 'scenes': selected, 'scenes_sha256': digest(selected),
            'allowed': True,
        }
        atomic_json(self.directory / 'outcome.json', outcome)
        return selected
