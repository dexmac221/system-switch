# System Switch

**When should a fast decision model stop and think?** Code and data of the paper *System Switch: When
Should a Fast Decision Model Stop and Think? A closed-loop study with typed-decision models and a
vision reasoner in Doom* (Gianluca Bailo, 2026), [arXiv:2610.09683](https://arxiv.org/abs/2610.09683)
([PDF](https://arxiv.org/pdf/2610.09683)).

A fast "System One" typed-decision actor plays Doom through
[doomLaya](https://github.com/azalio/doomLaya) and takes every decision. A gate watches the recent
history (low confidence, no progress, blocked commands, loops, no new area). When it opens, a reasoning
vision LLM is asked for one thought and a command, which then guides the actor as a plan while the game
keeps running in real time.

## Results in short

Offline results are on 900 held-out questions; 95% intervals resample games.

- **Zero-shot decision models over-choose collecting items.** Among their errors they collect 1.6–1.8
  times more often than chance, whatever the order of the options. The order does change some
  models' accuracy: Strands Decider goes from 0.59 to 0.38 on commands when the options are shuffled.
- **Accuracy, calibration and sensitivity are distinct.** Sensitivity here is the AUROC of confidence
  for right vs. wrong answers. The most sensitive actor (Laya base, 0.89) is at chance *within*
  categories: its confidence tracks which kinds of situation it fails, not which answers.
- **Deferring 30% of decisions to a reasoner (Qwen3.6-35B-A3B).** The gain over deferring at random
  follows the actor's AUROC: rank correlation 0.87 over 18 actor–option-order pairs. With actor and rate
  chosen on held-out games, the gain is:
  - +0.13 [0.08, 0.18] with doomLaya's option order;
  - +0.08 [0.02, 0.14] with shuffled options.

  Reasoning carries about half of it.
- **Closed loop (MAP01, 33 games, 3 seeds per variant): no exit.**
  - Committing to plans covers more of the map and opens more doors. This holds for the reasoner's
    plans and for a fixed "explore" rule behind the same gate alike, and it gets an actor that stands
    still on its own to play.
  - With the rule the agent dies more often (3.2 vs. 1.0 deaths per game over both actors).
  - Told that some doors need keys, the reasoner takes ordinary closed doors for locked ones.

## Contents

| path | what |
|---|---|
| `src/switch.py` | the System Switch: gate, thinker, plans (`--mode off/switch/rule/advise/narrate`, `--triggers`, `--basics`, `--commit`, `--no-image`, ...) |
| `src/pilot.sh` | one game through the switch, one JSON line out |
| `src/serve_systemone.py` | doomLaya `/predict` bridge to any llama.cpp `/v1/systemone` server |
| `src/serve_llm.py` | doomLaya `/predict` bridge to a chat LLM (doomGemma) |
| `src/bench_systemone.sh` | one System One model, zero-shot: server, bridge and evaluation |
| `src/evaluate.py` | accuracy, confidence and per-question records of an actor on a question file (`EVAL_DATA`) |
| `src/build_testset.py` | the held-out test set, with doomLaya's own oracle on unseen seeds |
| `src/eval_thinker.py`, `src/thinker_fallback.py` | a chat LLM as zero-shot decision maker, with or without reasoning |
| `src/paper_tables.py`, `src/review_analyses.py`, `src/stats_ci.py`, `src/deferral_curve.py` | every offline table and the figure: accuracy, ECE, AUROC overall and within category, collect bias, deferral, selection on held-out games |
| `src/summarize_games.py`, `src/thought_stats.py`, `src/loop_metrics.py`, `src/thoughts_to_srt.py` | the closed-loop games: outcomes, thoughts, coverage, closest approach to the exit, doors, thinker load |
| `patches/` | `doomLaya-agent.patch` (adds the `llm` model to doomLaya's agent), then `apply_switch_patch.py` (adds `--send-frame`) |
| `data/` | test set, per-question records of every actor and thinker, switch logs and per-tic positions of every game (see `data/README.md`) |

## Reproducing

1. **doomLaya**: commit `b25edd3`. Apply `git apply patches/doomLaya-agent.patch`, then
   `python patches/apply_switch_patch.py` from the doomLaya directory. Copy `src/` into doomLaya
   (e.g. as `switch/`).
2. **llama.cpp**: commit `cb7934c` or later, for `/v1/systemone`. Encoder models (Laya, Julia, Clef)
   need `-b 2048 -ub 2048`, otherwise the server aborts on long Doom states.
3. **Thinker**: Qwen3.6-35B-A3B served by llama.cpp with its vision projector (`--jinja`, reasoning
   on), e.g. on port 8090.
4. **Offline**:
   - `EVAL_DATA=data/test_set/test_900.json bash src/bench_systemone.sh <name> <model.gguf>` for each
     actor;
   - `eval_thinker.py --think --max-tokens 4000 --kind command` for the thinker;
   - then `paper_tables.py` and `review_analyses.py` on the records.

   The records in `data/` reproduce every table without running any model.
5. **Closed loop**: `src/pilot.sh <tag> <actor /predict URL> <actor model> <mode> MAP01 <seed> [switch args]`.
   The actor must run on a GPU. doomLaya does not wait for it: at about 2 s per decision a game has
   107 decisions instead of 350, and it plays differently.

## Citation

```bibtex
@misc{bailo2026systemswitch,
  title         = {System Switch: When Should a Fast Decision Model Stop and Think?},
  author        = {Gian Luca Bailo},
  year          = {2026},
  eprint        = {2610.09683},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2610.09683},
  url           = {https://arxiv.org/abs/2610.09683}
}
```

## License

Apache License 2.0; see `LICENSE` and `NOTICE` for the third-party work this builds on. The paper is on
arXiv under CC BY 4.0.
