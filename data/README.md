# Data of the System Switch paper

Everything the tables and the figure are computed from (`src/paper_tables.py`,
`src/review_analyses.py`, `src/summarize_games.py`, `src/thought_stats.py`).

| folder | content |
|---|---|
| `test_set/test_900.json` | the 900 held-out questions (600 command, 300 weapon), built with doomLaya's oracle on MAP01 games with seeds 45–53 (`src/build_testset.py`); each record has `state`, `question`, `label`, `kind`, `category`, `source_run`, `source_tick` |
| `test_set/test_900_shuffled_options.json` | the same questions with the options in a random order per question (seed = question index) |
| `actors/fixed_order/`, `actors/shuffled_options/` | one line per question and actor (`evaluate.py`): `i` (index in the test set), `kind`, `label`, `choice`, `conf` (top probability), `right` |
| `thinkers/` | one line per command question (`eval_thinker.py`): `choice`, `right`, completion `tokens`, `ms`; `*_think` with reasoning, `*_fast` without, `*_thinkfb` reasoning with the switch's fallback (`thinker_fallback.py`); fixed option order only |
| `games/games_g5.jsonl`, `games_r5.jsonl`, `games_q5.jsonl` | one line per closed-loop game (`pilot.sh`): variant, seed, kills, deaths, thoughts, share of decisions guided by a plan |
| `games/switch_logs/` | the switch's log of every game: each request (actor's choice, confidence, final choice, source), each thought (triggers, text, choice, latency, fallback) and each plan end |
| `games/runs/` | per game: doomLaya's `summary.json` (doors opened, latency, wall vs game seconds) and the player's position per tic (`seconds,x,y,hp`, gzip) |
| `tables/` | the text output of the analysis scripts and the deferral curve (CSV) |

Tags: `g5-*` and `r5-*` are the 30 games of Table 4 (MAP01, seeds 48–50, on an RTX 4070 machine); `q5-*` are the rule control with the base Laya and one check game, played on an RTX 2070 SUPER machine without video.
Actor names: `laya-typed` and `laya-base` are the two released Laya revisions through doomLaya's PyTorch
server, `laya-gguf` the base through llama.cpp. `qwen36uff` and `gemma4uff` are the official checkpoints
of Qwen3.6-35B-A3B and Gemma 4 26B-A4B-it.

Videos of the games are not included (several hundred MB).

## Added for System Switch II

| folder | content |
|---|---|
| `new_models/` | one line per question for each decision model released after the first paper (`src/eval_decision_models.py`); same fields as `actors/`, plus `x` with each backend's extras; versions, revisions and request settings in `new_models/VERSIONS.md` |
| `teacher/rounds/<round>/` | the four teaching rounds (`gemma1`, `gemma2`, `deepseek1`, `dagger1`; MAP01, seeds 100–111): `etichette_s<seed>.jsonl`, one line per label (field names in Italian, documented in `src/teacher_rounds.py`), and `progresso.log`, one line per game |
| `teacher/training/` | the fine-tune log of every student (`<run>.out`: agreement with the teacher on the held-out seeds before training and per epoch) and its data summary |
| `teacher/eval_games.jsonl` | one line per evaluation game: 12 variants × 18 seeds (48–65), MAP01, 180 s, the game waits for the model |
| `teacher/eval_decisions.jsonl.gz` | one line per decision in the evaluation games of ten variants (untrained, Laya-v3, the six 5-epoch students, the two second-round students): game time, command type chosen, probability of explore when offered, options, health, whether an enemy or a health item is in sight, and a short hash of the state text; extracted from the games' decision logs |
| `tables/system-switch-ii/` | the text output of the System Switch II scripts, as distinct questions (`*_distinct.txt`) and as released |

