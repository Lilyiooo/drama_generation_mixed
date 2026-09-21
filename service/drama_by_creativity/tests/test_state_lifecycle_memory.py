import asyncio
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import AsyncMock, patch

from drama_local.runtime import Context, LocalStore, atomic_json
from service.drama_by_creativity.state_lifecycle_memory import StateLifecycleMemory, digest, initial_evidence_catalog
from tools import state_hybrid_study as study
from tools import resume_state_hybrid as resume


generator = importlib.import_module("service.drama_by_creativity.generate_episode_script")


class StateLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {
            "DRAMA_OUTPUT_DIR": str(self.root), "DRAMA_STATE_LIFECYCLE_HYBRID": "1",
            "DRAMA_DISABLE_PLOT_RETRIEVAL": "1", "DRAMA_SUMMARY_MEMORY_BASELINE": "false",
            "DRAMA_ADJUST_WORD_COUNT": "false", "DRAMA_ADJUST_COHERENCE": "false",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.memory = StateLifecycleMemory("test", self.root)
        self.assets = {"world_view": "大雍的原始世界设定", "role_info": "萧宴：原始将军人设",
                       "story_outline": "原总纲", "episode_outlines": [{"episode_id": 0, "content": "开场大纲"}]}
        with patch.object(self.memory, "extract_initial_state", AsyncMock(return_value=self.initial_payload())):
            asyncio.run(self.memory.initialize(Context(), self.assets))

    def initial_payload(self):
        state = {field: [] for field in self.memory.engine.STATE_FIELDS}
        state["character_state"] = ["萧宴：将军"]
        return {"initial_state": state, "evidence": [{"field": "character_state", "text": "萧宴：将军",
                                                     "source": "role_info", "quote": "原始将军人设"}]}

    def extraction(self, **delta):
        state = {field: [] for field in self.memory.engine.STATE_FIELDS}
        state.update(delta)
        return {"state_delta": state, "resolved_goals": [], "resolved_unknown": [], "retracted_facts": []}

    def update(self, index, result, script=None):
        with patch.object(self.memory, "extract", AsyncMock(return_value=result)):
            return asyncio.run(self.memory.update_from_episode(Context(), index, f"大纲{index}", script or f"剧本{index}"))

    def test_lifecycle_cleans_resolved_items_and_retains_unresolved(self):
        self.update(0, self.extraction(current_goals=["解读欠条", "追查幕后主使"],
                                      unknown_information=["账本藏在哪里"], confirmed_facts=["账本在库房"]))
        result = self.extraction(known_information=["账本在书房"])
        result.update(resolved_goals=["解读欠条"], resolved_unknown=["账本藏在哪里"], retracted_facts=["账本在库房"])
        record = self.update(1, result)
        self.assertEqual(record["next_state"]["current_goals"], ["追查幕后主使"])
        self.assertEqual(record["next_state"]["unknown_information"], [])
        self.assertEqual(record["next_state"]["confirmed_facts"], [])
        self.assertEqual(StateLifecycleMemory("test", self.root).state, record["next_state"])
        with self.assertRaisesRegex(ValueError, "different script"):
            asyncio.run(self.memory.update_from_episode(Context(), 1, "大纲1", "被篡改的剧本"))

    def test_hybrid_respects_budget_without_truncating_stored_state(self):
        facts = [f"第{index}条线索：库房钥匙归属与欠条证据已记录，账房先生待核实证据" for index in range(600)]
        self.update(0, self.extraction(confirmed_facts=facts, current_goals=["找到账本钥匙"]))
        full_hash = digest(self.memory.state)
        context = self.memory.context(1, "调查库房钥匙和账本")
        decision = json.loads((self.memory.root / "retrieval/E02.json").read_text())
        self.assertLessEqual(decision["selection"]["selected_chars"], 6000)
        self.assertGreater(decision["selection"]["full_chars"], 6000)
        self.assertEqual(digest(self.memory.state), full_hash)
        self.assertIn("找到账本钥匙", context)
        with self.assertRaisesRegex(ValueError, "requires"):
            self.memory.context(3, "跳过中间集")

    def test_extraction_retry_keeps_previous_state_and_script(self):
        self.update(0, self.extraction(current_goals=["查清账房身份"]))
        request = AsyncMock(side_effect=[SimpleNamespace(response="{}"),
                                        SimpleNamespace(response=json.dumps(self.extraction(), ensure_ascii=False))])
        with patch("service.drama_by_creativity.state_lifecycle_memory.LLM.request", request):
            asyncio.run(self.memory.extract(Context(), 1, "本集完整正文"))
        self.assertEqual(request.await_count, 2)
        for call in request.await_args_list:
            self.assertIn("查清账房身份", call.args[1]["Prompt"])
            self.assertIn("本集完整正文", call.args[1]["Prompt"])

    def prepare_study(self):
        source = self.root / "source"
        episodes = [{"season_id": 0, "episode_id": index, "content": f"第{index + 1}集大纲"} for index in range(60)]
        source_data = {
            "pipeline_result.json": {
                "script_proposal": {"story_outline": {"role_info": ["萧宴：原始将军人设"],
                                                       "world_building": ["大雍的原始世界设定"], "story_outline": []}},
                "story_outline": {"story_outline": {"role_info": [], "world_building": [], "story_outline": ["原总纲"]}},
                "episode_outline": {"episode_outline": {"seasons": [{"season_id": 0, "episodes": episodes}]}},
            },
            "episode_outlines.json": [{"episode_id": index + 1} for index in range(60)],
            "01_script_proposal/01_world_view.json": {"world_view": "大雍的原始世界设定"},
            "01_script_proposal/02_role_setting.json": [{"name": "萧宴", "role": "原始将军人设"}],
            "04_future_map/future_map.json": {"nodes": [{"node_id": "FM001", "goal": "找到真相"}]},
            "04_future_map/episode_contributions.json": {"episodes": []},
        }
        for name, value in source_data.items():
            atomic_json(source / name, value)
        atomic_json(source / "04_future_map/future_driving_graph.json", {"nodes": ["不应复制的旧剧本记忆"]})
        experiment = self.root / "study"
        study.prepare(source, experiment)
        return experiment

    def test_fixed_inputs_restore_bible_and_exclude_source_history(self):
        experiment = self.prepare_study()
        inputs = study.load_inputs(experiment)
        request = study.make_request(inputs, "hybrid", 0)
        self.assertEqual(request.generate_data.story_outline.world_building, ["大雍的原始世界设定"])
        self.assertIn("原始将军人设", request.generate_data.story_outline.role_info[0])
        for arm in study.ARMS:
            self.assertFalse((experiment / arm / "04_future_map/future_driving_graph.json").exists())
            self.assertFalse((experiment / arm / "05_drama").exists())
        study.prepare(self.root / "source", experiment)

    def test_memory_and_bible_reach_both_generation_stages(self):
        experiment = self.prepare_study()
        directory = experiment / "hybrid"
        study.configure(directory, "hybrid")
        scene = AsyncMock(return_value=[{"场次编号": "1-1", "情节概要": "找到账本"}])
        request = AsyncMock(return_value=SimpleNamespace(response="1-1. 库房 日\n萧宴：找到账本了。"))
        native_update = AsyncMock(return_value={"future_driving_nodes": [], "future_map_edges": [], "world_state_updates": []})
        with patch.object(generator, "scene_outline_inference_json", scene), \
             patch.object(generator.LLM, "request", request), \
             patch.object(generator.NarrativeMemory, "update_from_episode", native_update), \
             patch.object(StateLifecycleMemory, "extract_initial_state", AsyncMock(return_value=self.initial_payload())), \
             patch.object(StateLifecycleMemory, "extract", AsyncMock(return_value=self.extraction())):
            asyncio.run(generator.generate_episode_script(
                Context(), study.make_request(study.load_inputs(experiment), "hybrid", 0), LocalStore()))
        scene_params = scene.await_args.kwargs["params"]
        script_params = request.await_args.args[1]
        self.assertIn("状态生命周期记忆", scene_params["NarrativeMemory"])
        self.assertEqual(scene_params["NarrativeMemory"], script_params["PrevAbs"])
        for params in (scene_params, script_params):
            self.assertEqual(params["WorldView"], "大雍的原始世界设定")
            self.assertIn("原始将军人设", params["RoleInfo"])
        self.assertIn("萧宴：将军", scene_params["NarrativeMemory"])
        self.assertTrue((directory / "generation_assets.json").exists())
        self.assertEqual(len(StateLifecycleMemory(study.story_id("hybrid")).records), 1)

    def test_resume_extracts_saved_script_without_generating_again(self):
        experiment = self.prepare_study()
        directory = experiment / "hybrid"
        study.configure(directory, "hybrid")
        atomic_json(directory / "05_drama/episode_01.json", {"episode_id": 0, "content": "已保存的剧本"})
        native = study.NarrativeMemory(study.story_id("hybrid"))
        native.data["last_updated_episode"] = 0
        native._save()
        generate = AsyncMock()
        with patch.object(study, "verify_server"), patch.object(study, "generate_episode_script", generate), \
             patch.object(StateLifecycleMemory, "extract_initial_state", AsyncMock(return_value=self.initial_payload())), \
             patch.object(StateLifecycleMemory, "extract", AsyncMock(return_value=self.extraction())):
            asyncio.run(study.generate(experiment, "hybrid", 1))
        generate.assert_not_awaited()
        self.assertEqual(study.progress(experiment, "hybrid")["complete"], 1)
        self.assertEqual(json.loads((directory / "05_drama/episode_01.json").read_text())["content"], "已保存的剧本")

    def test_control_uses_bible_without_added_state_memory(self):
        experiment = self.prepare_study()
        study.configure(experiment / "native_context", "native_context")
        scene = AsyncMock(return_value=[{"场次编号": "1-1", "情节概要": "原生分场"}])
        native_update = AsyncMock(return_value={"future_driving_nodes": [], "future_map_edges": [], "world_state_updates": []})
        with patch.object(generator, "scene_outline_inference_json", scene), \
             patch.object(generator.LLM, "request", AsyncMock(return_value=SimpleNamespace(response="原生正文"))), \
             patch.object(generator.NarrativeMemory, "update_from_episode", native_update), \
             patch.object(generator, "StateLifecycleMemory") as memory:
            asyncio.run(generator.generate_episode_script(
                Context(), study.make_request(study.load_inputs(experiment), "native_context", 0), LocalStore()))
        memory.assert_not_called()
        self.assertNotIn("状态生命周期记忆", scene.await_args.kwargs["params"]["NarrativeMemory"])
        self.assertIn("原始将军人设", scene.await_args.kwargs["params"]["RoleInfo"])

    def test_initial_state_is_grounded_and_bound_to_generated_assets(self):
        with patch.object(self.memory, "extract_initial_state", AsyncMock()) as extraction:
            asyncio.run(self.memory.initialize(Context(), self.assets))
            extraction.assert_not_awaited()
        changed = {**self.assets, "role_info": "换了一套人物"}
        with self.assertRaisesRegex(ValueError, "different generation"):
            asyncio.run(self.memory.initialize(Context(), changed))
        invalid = self.initial_payload()
        invalid["evidence"][0]["quote"] = "源资产里根本不存在的事实"
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.memory.validate_initial_result(invalid, self.assets)
        invalid["evidence"][0].update(source="episode_outlines", quote="开场大纲")
        with self.assertRaisesRegex(ValueError, "generated world_view or role_info"):
            self.memory.validate_initial_result(invalid, self.assets)

    def test_initializer_receives_real_bible_and_only_opening_episode_context(self):
        assets = {**self.assets, "episode_outlines": [*self.assets["episode_outlines"],
                                                      {"episode_id": 59, "content": "六十集之后才发生的未来事件"}]}
        entries = {"entries": [{"field": "character_state", "text": "萧宴：将军", "evidence_ids": ["R0001"]}]}
        client = AsyncMock(return_value=SimpleNamespace(response=json.dumps(entries, ensure_ascii=False)))
        with patch("service.drama_by_creativity.state_lifecycle_memory.LLM.request", client):
            result = asyncio.run(self.memory.extract_initial_state(Context(), assets))
        prompt = client.await_args.args[1]["Prompt"]
        self.assertIn("原始将军人设", prompt)
        self.assertIn("大雍的原始世界设定", prompt)
        self.assertIn("原总纲", prompt)
        self.assertIn("开场大纲", prompt)
        self.assertNotIn("六十集之后才发生的未来事件", prompt)
        self.assertEqual(result["initial_state"], self.initial_payload()["initial_state"])
        self.assertEqual(result["evidence"][0]["quote"], self.assets["role_info"])
        self.memory.validate_initial_result(result, assets)

    def test_evidence_ids_resolve_to_unmodified_source_quotes(self):
        assets = {**self.assets, "role_info": "#### 萧宴\n身中奇毒“蚀骨散”。背负巨债；流放途中被救！"}
        catalog = initial_evidence_catalog(assets)
        self.assertEqual(list(catalog), ["W0001", "R0001", "R0002", "R0003", "R0004"])
        for item in catalog.values():
            self.assertIn(item["quote"], assets[item["source"]])
        result = self.memory.resolve_initial_entries({"entries": [
            {"field": "character_state", "text": "萧宴身中奇毒且负债", "evidence_ids": ["R0002", "R0003", "R0002"]}
        ]}, catalog)
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(result["evidence"][0]["quote"], "身中奇毒“蚀骨散”。")
        self.memory.validate_initial_result(result, assets)

    def test_initializer_rejects_invented_ids_and_oversized_state(self):
        catalog = initial_evidence_catalog(self.assets)
        entry = {"field": "character_state", "text": "萧宴：将军", "evidence_ids": ["R9999"]}
        with self.assertRaisesRegex(ValueError, "unknown evidence_id"):
            self.memory.resolve_initial_entries({"entries": [entry]}, catalog)
        with self.assertRaisesRegex(ValueError, "1 to 32"):
            self.memory.resolve_initial_entries({"entries": [entry] * 33}, catalog)
        with self.assertRaisesRegex(ValueError, "strings"):
            self.memory.resolve_initial_entries({"entries": [{**entry, "evidence_ids": [{}]}]}, catalog)
        with self.assertRaisesRegex(ValueError, "only entries"):
            self.memory.resolve_initial_entries(self.initial_payload(), catalog)

    def test_initializer_retry_identifies_bad_id_and_preserves_failure_history(self):
        entry = {"field": "character_state", "text": "萧宴：将军", "evidence_ids": ["R0001"]}
        bad = json.dumps({"entries": [{**entry, "evidence_ids": ["R9999"]}]})
        good = json.dumps({"entries": [entry]})
        request = AsyncMock(side_effect=[SimpleNamespace(response=bad), SimpleNamespace(response=good),
                                         SimpleNamespace(response=good)])
        old_failure = self.memory.root / "failures/initial_state_2.json"
        atomic_json(old_failure, {"response": "old quote failure"})
        with patch("service.drama_by_creativity.state_lifecycle_memory.LLM.request", request):
            asyncio.run(self.memory.extract_initial_state(Context(), self.assets))
            asyncio.run(self.memory.extract_initial_state(Context(), self.assets))
        retry_prompt = request.await_args_list[1].args[1]["Prompt"]
        self.assertIn("R9999", retry_prompt)
        self.assertIn("原始将军人设", retry_prompt)
        self.assertIn("原总纲", retry_prompt)
        attempts = list((self.memory.root / "initialization_attempts").iterdir())
        self.assertEqual(len(attempts), 2)
        errors = [study.read_json(path) for path in (self.memory.root / "initialization_attempts").glob("*/attempt_*.json")]
        self.assertTrue(any(item.get("response") == bad and "error" in item for item in errors))
        self.assertEqual(study.read_json(old_failure), {"response": "old quote failure"})

    def prepare_live_resume(self):
        experiment = self.prepare_study()
        inputs = study.load_inputs(experiment)
        directory = experiment / "hybrid"
        assets = study.assets_from_request(study.make_request(inputs, "hybrid", 0, "APID-test-001"))
        atomic_json(directory / "generation_assets.json", assets)
        atomic_json(directory / "state_lifecycle/APID-test-001/story_assets.json", assets)
        return directory

    def test_live_resume_uses_exact_saved_assets_without_pipeline_result(self):
        directory = self.prepare_live_resume()
        self.assertFalse((directory / "pipeline_result.json").exists())
        inputs = resume.saved_inputs(directory, "APID-test-001")
        actual = study.assets_from_request(study.make_request(inputs, "hybrid", 0, "APID-test-001"))
        self.assertEqual(actual, study.read_json(directory / "generation_assets.json"))
        with self.assertRaisesRegex(ValueError, "original state assets"):
            resume.saved_inputs(directory, "wrong-story-id")
        changed = {**actual, "world_view": "不是本轮的世界观"}
        atomic_json(directory / "generation_assets.json", changed)
        with self.assertRaisesRegex(ValueError, "original state assets"):
            resume.saved_inputs(directory, "APID-test-001")

    def test_live_resume_repairs_saved_script_using_original_story_id(self):
        directory = self.prepare_live_resume()
        inputs = resume.saved_inputs(directory, "APID-test-001")
        study.configure(directory, "hybrid")
        native = study.NarrativeMemory("APID-test-001")
        native.data["last_updated_episode"] = 0
        native._save()
        atomic_json(directory / "05_drama/episode_01.json", {"episode_id": 0, "content": "本轮已保存正文"})
        generate = AsyncMock()
        with patch.object(study, "verify_server"), patch.object(study, "generate_episode_script", generate), \
             patch.object(StateLifecycleMemory, "extract_initial_state", AsyncMock(return_value=self.initial_payload())), \
             patch.object(StateLifecycleMemory, "extract", AsyncMock(return_value=self.extraction())), redirect_stdout(StringIO()):
            asyncio.run(study.generate_directory(inputs, directory, "hybrid", "APID-test-001", 1))
            asyncio.run(study.generate_directory(inputs, directory, "hybrid", "APID-test-001", 1))
        generate.assert_not_awaited()
        self.assertEqual(study.directory_progress(directory, "hybrid", "APID-test-001")["complete"], 1)
        self.assertFalse((directory / "state_lifecycle/state-hybrid-v2-hybrid").exists())

    def test_live_pipeline_passes_generated_bible_into_and_out_of_outline_stage(self):
        from drama_local import models as pb
        from service.drama_by_creativity.tests import run_test
        outline_module = importlib.import_module("service.drama_by_creativity.generate_story_outline_and_role")
        proposal = pb.GenerateScriptProposalRsp(story_outline=pb.StoryOutline(
            world_building=[self.assets["world_view"]], role_info=[self.assets["role_info"]]))
        outline_json = {"title": "测试故事", "background": "故事背景", "framework": [], "highlights": "亮点", "summary": "总纲"}
        async def create_outline(ctx, request):
            self.assertEqual(request.generate_input.world_view, self.assets["world_view"])
            self.assertEqual(request.generate_input.role_setting, self.assets["role_info"])
            return await outline_module.generate_outline_by_script(ctx, request, None)
        args = SimpleNamespace(story_id="test", plot_type=pb.PlotType.SHORT_CARTOON, episode_nums=60)
        service = SimpleNamespace(GenerateStoryOutline=create_outline)
        with patch.object(outline_module, "llm_inference_json", AsyncMock(return_value=outline_json)), \
             patch.object(outline_module, "refine_story_outline_eventlines", AsyncMock(return_value=outline_json)), \
             redirect_stdout(StringIO()):
            result = asyncio.run(run_test.step_3_story_outline(Context(), service, args, proposal))
        self.assertEqual(result.story_outline.world_building, proposal.story_outline.world_building)
        self.assertEqual(result.story_outline.role_info, proposal.story_outline.role_info)
        persisted = json.loads((self.root / "03_story_outline/story_assets.json").read_text())
        self.assertEqual(persisted, result.story_outline.to_dict())


if __name__ == "__main__":
    unittest.main()
