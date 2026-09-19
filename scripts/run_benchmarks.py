"""Run the full benchmark suite and print a Markdown table for the README."""
import json
from pathlib import Path

from jev_reranker.eval_synthetic import generate_cases, run_eval

cases = generate_cases(300, 7)
metrics = run_eval(cases)
out = Path("benchmarks/results/synthetic_eval.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
