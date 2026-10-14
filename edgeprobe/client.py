"""Minimal client for a local llama.cpp server (OpenAI-compatible chat endpoint)."""
import math
import os
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = os.environ.get("LLAMA_SERVER", str(ROOT / ".cache/src/build/bin/llama-server"))


class LlamaServer:
    """Starts llama-server for one GGUF model and exposes chat()."""

    def __init__(self, model_path, port=8088, threads=2, ctx=2048, binary=DEFAULT_BIN):
        self.url = f"http://127.0.0.1:{port}"
        self.log = open(ROOT / ".cache" / f"server_{Path(model_path).stem}.log", "w")
        self.proc = subprocess.Popen(
            [binary, "-m", str(model_path), "--port", str(port), "-t", str(threads),
             "-c", str(ctx), "--parallel", "1", "--jinja", "--no-webui"],
            stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(600):
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited; see " + self.log.name)
            time.sleep(1)
        raise RuntimeError("llama-server did not become healthy")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def chat(self, messages, max_tokens=64, temperature=0.0, seed=0, top_logprobs=0,
             top_p=1.0, top_k=0):
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "seed": seed, "top_p": top_p, "top_k": top_k, "min_p": 0.0,
                "cache_prompt": True}
        if top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        for attempt in range(5):
            try:
                r = requests.post(self.url + "/v1/chat/completions", json=body, timeout=600)
                r.raise_for_status()
                ch = r.json()["choices"][0]
                text = ch["message"]["content"] or ""
                lp = (ch.get("logprobs") or {}).get("content") or []
                return text.strip(), lp
            except (requests.RequestException, KeyError) as e:
                err = e
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"chat failed: {err}")


def binary_prob(top, yes_words, no_words):
    """P(yes) renormalised over yes/no tokens found among the first token's top-k."""
    py = pn = 0.0
    for t in top:
        w = t["token"].strip().lower()
        p = math.exp(t["logprob"])
        if w in yes_words:
            py += p
        elif w in no_words:
            pn += p
    if py + pn == 0:
        return None
    return py / (py + pn)
