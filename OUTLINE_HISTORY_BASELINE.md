# Direct-outline/all-history baseline

This experiment uses the exact frozen `generation_assets.json` shared by all six
trajectories in `full_scene_gate_ab_v1`. For episode N, Qwen3.6 receives the
world view, character settings, story outline, current episode outline, and the
verbatim scripts generated for baseline episodes 1 through N-1.

The baseline deliberately excludes scene-outline generation, narrative memory,
state lifecycle memory, obligations, hybrid retrieval, Strategy cards, state
gates, plot-library retrieval, and post-generation rewriting.

References to `candidate_gate` and `control_hybrid` in the driver are limited to
verifying that all trajectories share the same frozen generation assets and to
reading completed scores for the final comparison report. The baseline
generation path does not execute either method or import their components.

R01, R02, and R03 are independent stochastic runs. Episodes remain sequential
within each run because episode N depends on every earlier script. The three runs
may execute concurrently.

```bash
cd /path/to/drama_generation_mixed
export GENERATION_MODEL_PATH=/path/to/Qwen3.6-27B
bash outline_history_baseline.sh prepare
bash outline_history_baseline.sh generate --workers 3 --execute-api
bash outline_history_baseline.sh status
bash outline_history_baseline.sh evaluate --workers 3 --execute-api
bash outline_history_baseline.sh report
```

All outputs are written under
`output/outline_direct_all_history_baseline_v1`. Re-running `generate` or
`evaluate` resumes completed work. Every accepted episode records the complete
history list, per-episode history hashes, prompt hash, exact prompt token count,
seed, output hash, and API usage in `request_metadata`.

The default source is `output/full_scene_gate_ab_v1`, whose six trajectories
must contain identical `generation_assets.json` files. Set
`BASELINE_SOURCE_ROOT` to use the same frozen assets from another location and
`BASELINE_OUTPUT_ROOT` to move baseline outputs. `SCRIPT_ROOT` changes both
defaults together. Set `DRAMA_EVALUATOR_ROOT` only when the evaluator is not in
this repository's `evaluation/drama_evaluator_logic_quality` directory.
