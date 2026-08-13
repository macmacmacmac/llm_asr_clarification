# Agent guide

Guidance for any AI coding agent (Claude Code, Antigravity CLI, Cursor, Codex, …) working in this
repository. `AGENTS.md` is the real file; `CLAUDE.md` is a symlink to it, so whichever name your
agent looks for, it reads this content. Edit `AGENTS.md`; do not replace the symlink with a second
copy that will drift.

Everything below is about *this repo*, not about any particular agent, and applies verbatim
regardless of which tool is reading it.

## IMPORTANT: never execute scripts

This is a Slurm-cluster project. **Do not run any of the scripts in this repo** — no `run_scripts.py`, no `python -m ...`, no `sbatch`/`srun`, no notebook execution, not even "quick" one-off invocations to check behavior. Jobs consume shared cluster resources, hit paid APIs, and write into the shared dataset directory.

Verifying that a script actually runs correctly is the **user's** responsibility. After making changes, the only check to perform is a compile check:

```bash
python -m py_compile <changed_file.py>
```

Report the result of that and stop; hand any runtime verification back to the user with the exact command they should run.

## No walkthrough artifacts

Do **not** create `walkthrough.md` or similar summary artifacts after completing a task. They waste tokens and provide no value — the diff and conversation history are sufficient. Just state what you changed and stop.

## What this project is

A research codebase studying whether *clarifying* mistranscribed ASR segments improves downstream comprehension of meeting transcripts. The AMI meeting corpus is transcribed with Whisper, a detector flags likely-mistranscribed lines, "important" lines among those are replaced with ground truth (simulating a clarification request), and the resulting transcript is evaluated by having an LLM answer a quiz generated from the ground-truth transcript.

It is a Slurm-cluster research project, not a deployable product: there is no test suite, no linter config, and scripts are frequently edited in place with alternative approaches left commented out.

## Environment & commands

Dependencies are managed with `uv` (`pyproject.toml` + `uv.lock`, Python ≥3.10, torch from a CUDA 12.8 index). The checked-in `.venv` is what job scripts activate.

```bash
uv sync                       # install/update deps
uv run python run_scripts.py --scripts <module.path> [script args]
```

Secrets live in `.env` at the project root (`OPENAI_API_KEY`, `HF_TOKEN`), loaded by `OpenAIWrapper` via `dotenv`.

### Running scripts

`run_scripts.py` is the single entry point. It takes module paths relative to `llm_asr_clarification.scripts`, imports each, and calls its `run(unknown_args)`. Everything after `--scripts` that it doesn't recognize is forwarded verbatim to every named script — this is how multiple stages get chained in one job:

```bash
python run_scripts.py --scripts clarifications.pipelinev2 --dataset-path ./shared/datasets/amicorpus/train
python run_scripts.py --scripts quiz.question_answerer quiz.question_scorer \
    --transcript_file custom_transcript_gt_segments --question_file parsed_diarized_gt --do_all_meetings
```

### Slurm

`cpu_job.sh` and `gpu_job.sh` are thin wrappers (`srun python -m run_scripts "$@"`) — pass script args straight through. `run_job.sh` is a standalone A100 job with the command hardcoded at the bottom.

```bash
sbatch -t 300 -o out.log -e out.log cpu_job.sh --scripts quiz.question_scorer --do_all_meetings
sbatch gpu_job.sh --scripts transcription.custom.generate_diarized_gt_segments --dataset-path ./shared/datasets/amicorpus/validation
```

`run_quizpipeline.py` is a scratch driver that fans out one `sbatch` per (transcript, question-file) pair; edit its `transcript_files` list and `MODEL_TO_USE` rather than passing args.

`useful_commands.md` holds working invocations and the `setfacl` incantations for sharing the group dataset directory. Some commands there are stale (e.g. `quiz_pipeline.*` — the package is now `quiz.*`).

## Architecture

### Script convention

Every runnable module under `scripts/` exposes `run(args_list=None)` and follows the same preamble: `argparse` with `parse_known_args(args_list)`, `logger = get_logger(os.path.basename(__file__))`, then a log of every received arg. Cross-stage state is deliberately passed through **files on disk**, not function returns — new stages should read and write the meeting-folder layout below rather than importing each other.

`get_logger` writes to `./logs/<script>_<timestamp>.log` (DEBUG) and stdout (INFO). The `logs/` directory must exist or the script dies on import of the handler.

### Data layout (the real interface between stages)

Datasets live under the `shared/` symlink → `/group/jrwhitehill/llm_asr_clarification/shared/`, which is gitignored. Splits are `amicorpus/{train,validation,test}/<MEETING>/`:

```
<MEETING>/
  audio/          <MEETING>.Mix-Headset.wav, .Headset-N.wav, .Array*.wav
  transcripts/    parsed_diarized_gt.txt, custom_transcript_gt_segments.txt,
                  <base>_<detector>_clarify*.txt, whisper_tiny_diarized_transcript.txt, ...
  quiz/           quiz_from_<question_file>.json, failed_quiz_by_*.md
  artifacts/      beam_results_2.json, log_probs_*.csv
  summary.txt/.md
```

Transcript line format, parsed everywhere by regex on the first two integers:

```
(START - END)[Speaker <id>]: text
```

`beam_results_2.json` is the detector feature store: one entry per GT segment with `gt` plus `beam_1..beam_5`, each `{text, asr_avg_log_prob, llm_avg_log_prob}`. Every `MistranscriptionDetector` re-reads and re-flattens this file in its constructor, so line indices in a transcript, in the GT transcript, and in `beam_results_2.json` are all assumed to be **row-aligned**. Anything that changes segmentation breaks that assumption silently.

Quiz JSONs accumulate columns across runs: a list of `{question, correct_answer}` gains `answer_using_<transcript_file>` and `score_using_<transcript_file>` per transcript variant evaluated. Because many `sbatch` jobs write the same quiz file concurrently, `question_answerer` and `question_scorer` guard writes with `FileLock`.

### Pipeline stages

1. **Parse ground truth** — `scripts/parser/word_level_diarized.py` walks the AMI `words/*.xml`, merges per-speaker word queues into sentence-terminated segments, and writes `parsed_diarized_gt.txt` into the right split folder.
2. **Transcribe** — `scripts/transcription/{custom,whisper,qwen}/`. The one in active use is `custom/generate_diarized_gt_segments.py`: it segments the mix audio using GT timestamps, beam-searches Whisper (`num_return_sequences = num_beams`), scores each beam under a causal LLM (Llama-3.1-8B) as well as the ASR, does ECAPA-TDNN speaker attribution against per-headset enrollment embeddings, and emits both `custom_transcript_gt_segments.txt` (beam 1) and `beam_results_2.json` (all beams). `_vad_segments` / `_whisper_segments` variants differ only in how segment boundaries are chosen.
3. **Detect + clarify** — `scripts/clarifications/pipelinev2.py` is the current pipeline. For each meeting it runs a detector to get a mistranscription mask, computes "important" lines by embedding gold quiz answers and GT segments with `all-MiniLM-L6-v2` and taking top-k cosine matches, intersects the two sets, and replaces those lines with GT text. Output: `<base>_<detector>_clarify2.txt`. `pipeline.py` is the older v1 (ambiguity-pair based, `--strategy RANDOM|LLM-ORIG-CTX|LLM-GT-CTX`) and is kept for reference.
4. **Quiz** — `scripts/quiz/question_generator.py` (GPT generates N Q/A pairs from a transcript) → `question_answerer.py` (answers those questions given a candidate transcript) → `question_scorer.py` (GPT grades each answer 0/1 against the gold answer). Comparing mean `score_using_*` across transcript variants is the headline metric.
5. **Audit / analysis** — `clarifications/audit.py` produces per-meeting markdown of questions the GT answered but Whisper failed, with the relevant transcript excerpts. `scripts/score_transcripts.py` computes corpus WER. Notebooks in `src/llm_asr_clarification/` (`log_prob_beams_elbowplot.ipynb`, `quiz_analysis.ipynb`, `stopword_analysis.ipynb`, …) hold the threshold-tuning and result analysis, and are where `shared/model_weights/rf_model2.pkl` was trained.

### Detectors

`models/MistranscriptionDetector.py` defines the ABC (`pred_mistranscribed(line_numbers) -> list[bool]`) and three implementations, registered in `pipelinev2.CLS_MAP` under `RANDOM` / `RF` / `GT`:

- `GTDetector` — teacher/oracle: flags lines where ROUGE-L(hypothesis, gt) < 0.5.
- `RandomBernoulliDetector` — baseline, p = 0.592 (the measured mistranscription rate of whisper-tiny on train).
- `RFDetector` — random forest over aggregated per-line beam statistics (max/min/spread of ASR and LLM log probs), threshold 0.3745.

All three additionally require ≥5 Llama tokens in the line, so short backchannels are never flagged. Note that this length mask is a full-length Series ANDed against a `line_numbers`-indexed prediction — it only lines up when called with the complete range of lines.

`models/OpenAIWrapper.py` wraps chat completions (default `gpt-4o-mini`, `temperature=0`, retry on rate limit) and strips the `cookie` header via an httpx event hook — that hook works around a proxy on the cluster; don't remove it. Prompts live in `constants/{quiz,clarification,ambiguity_detection}_prompts.py`.

## Gotchas

- **Dataset path defaults are inconsistent and mostly stale.** Some scripts default to `./datasets/amicorpus/...` (a path that no longer exists) and others to `./shared/datasets/amicorpus/...`. Always pass `--dataset-path` / `--ami_path` explicitly. Note the flag name itself also varies between `--dataset-path` and `--ami_path` depending on the script's vintage.
- Scripts take a `--transcript_file` / `--question_file` **without** the extension in the quiz stage, but **with** `.txt` in the clarification stage.
- `filelock`, `pandas`, `tqdm`, and `scikit-learn` are used but not declared in `pyproject.toml`; they arrive transitively. Adding an import of them is not a new dependency in practice, but declaring one is a real fix.
- `ipdb.set_trace()` calls are scattered around commented out; leaving one live will hang a Slurm job.
- `constants/__init__.py` is empty, but older notebooks still `from llm_asr_clarification.constants import SAMPLE_MEETINGS`. Those cells no longer run.
- `scripts/archived/` is dead code kept for reference — don't wire it into anything.
