#!/usr/bin/env bash
# Run the full EdgeProbe protocol for every model, sequentially (CPU-only, resumable).
# Re-running the script only processes questions that are not yet in results/raw/<tag>.jsonl.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=.cache/models

run() {  # run <gguf> <tag>
  if [ -f "$M/$1" ]; then
    $PY -u scripts/run_model.py --model "$M/$1" --tag "$2" || echo "[$2] FAILED"
  else
    echo "[$2] SKIPPED (missing $M/$1)"
  fi
}

run qwen2.5-0.5b-instruct-q4_k_m.gguf   qwen2.5-0.5b
run qwen2.5-1.5b-instruct-q4_k_m.gguf   qwen2.5-1.5b
run Llama-3.2-1B-Instruct-Q4_K_M.gguf   llama3.2-1b
run qwen2.5-3b-instruct-q4_k_m.gguf     qwen2.5-3b
run Llama-3.2-3B-Instruct-Q4_K_M.gguf   llama3.2-3b
echo "ALL MODELS DONE"
