# Models and software behind the records in this folder (run on 25 November 2026)

One machine: Xeon E5-2673 v3, RTX 4070 (12 GB) + RTX 3060 (12 GB), Ubuntu 24.04, NVIDIA driver 595.

| records | model (Hugging Face repository, revision) | served with |
|---|---|---|
| `startlux4b_*.jsonl` | `startlux-models/StartLux-Decision-v1.1-4B-AutoThink-GGUF` (dbff9a6), file `...-Q8_0.gguf`, sha256 31e0d4e3...8213; tokenizer and configs from `StartLux-Decision-v1.1-4B-AutoThink` (23dd4c5) | llama.cpp `llama-server` at commit e8906e6b0 (`-ngl 99 -fa on --parallel 4 -c 32768`) behind `startlux-decide serve --backend llamacpp` (StartLuxLabs/StartLux-Decision, commit a99977d); four requests in flight |
| `startlux9b_*.jsonl` | `startlux-models/StartLux-Decision-v1.1-9B-AutoThink-GGUF` (f155faf), file `...-Q8_0.gguf`, sha256 f9525e9c...64fc; configs from `StartLux-Decision-v1.1-9B-AutoThink` (efc649d) | as above |
| `pplx_flash_v13.jsonl` | `perplexity-ai/pplx-decider-flash-v1.3` (d06fb3f) | the reference code in the repository (`source/src/pplx_decider`, version 0.5.0), `Decider.from_pretrained(dir, device="cuda")`, default mode; torch 2.14.0+cu130, transformers 5.17.0 |
| `laya05_vega_08b.jsonl`, `laya05_vega_4b.jsonl` | `nandakishorm/vega-U` (6aee375) with the readers `Qwen/Qwen3.5-0.8B` (2fc0636) and `Qwen/Qwen3.5-4B` (851bf6e) | `vegaml` 0.19.0 (`vegaml.load()` and `vegaml.load("4b")`), installed with `laya` 0.5.3; shipped calibration, nothing fitted |
| `laya05_classic_typed.jsonl`, `laya05_classic_base.jsonl` | `convaiinnovations/laya` (fd4f9db), subfolder `typed-decisions` and repository root | `laya` 0.5.3, `laya.load(...)` |

Request settings for StartLux: `think_mode` `never` (files `*_never*`) or `always` with `think_budget` 4096 (files
`*_always_cmd`), `allow_none` false, `return_details` true. With `allow_none` false the first read-out still lists a probability for
"None of the above" (`x.details.first_pass`), but that option cannot be chosen: `x.probs` and `conf` are
renormalized over the real options. `x.confidence` is the confidence the server reports, kept as is and not used. `*_never_shuffled` uses the test set with the options
shuffled per question (`test_900_shuffled_options.json` of the first paper).

Each line: `i` (index in the test set), `kind`, `label`, `choice`, `conf` (top probability), `right`, and `x` with
the extras of each backend (probabilities, latency, abstain flag and prediction set for vega-U, read-outs and thought
statistics for StartLux). Written by `src/eval_decision_models.py`.
