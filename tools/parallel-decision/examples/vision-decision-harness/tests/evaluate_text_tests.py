#!/usr/bin/env python3
"""
Evaluation runner for the text-based decision test suite.

Processes all .txt files in tests/text_samples/ through the llama-server
/decision endpoint using the shared sentiment schema, then compares results
against answer_key.json.

Usage:
    python3 evaluate_text_tests.py

Environment:
    LLAMA_SERVER_URL - defaults to http://0.0.0.0:8081
"""

import json
import os
import sys
import time
import requests

SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://0.0.0.0:8081")
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "text_samples")


def load_answer_key():
    path = os.path.join(SAMPLES_DIR, "answer_key.json")
    with open(path) as f:
        return json.load(f)


def run_single_decision(context_text, schema, instructions, seed=42):
    """Send a single decision request to the llama-server."""
    body = {
        "contexts": [context_text],
        "schema": schema,
        "instructions": instructions,
        "seed": seed,
    }
    resp = requests.post(f"{SERVER_URL}/decision", json=body, timeout=120)
    return resp


def main():
    key_data = load_answer_key()
    schema = key_data["schema"]
    question = key_data["question"]
    tests = key_data["tests"]
    expected_field = list(schema["properties"].keys())[0]  # "sentiment"

    print(f"Text Decision Test Suite")
    print(f"Server: {SERVER_URL}")
    print(f"Question: {question}")
    print(f"Schema: {json.dumps(schema['properties'])}")
    print(f"Tests: {len(tests)}")
    print()

    # Check server is reachable
    try:
        requests.get(f"{SERVER_URL}/v1/models", timeout=5)
    except Exception:
        print(f"ERROR: Cannot reach llama-server at {SERVER_URL}")
        print("The server must be running for tests to execute.")
        return 1

    instructions = "Classify the sentiment expressed in the following text."
    results = []
    t_total = time.time()

    for i, test in enumerate(tests):
        filepath = os.path.join(SAMPLES_DIR, test["file"])
        try:
            with open(filepath, "r") as f:
                text = f.read().strip()
        except Exception as e:
            results.append({"file": test["file"], "error": str(e), "correct": False})
            print(f"[{i+1:2d}/{len(tests)}] {test['file']:<8} ERROR: {e}")
            continue

        t0 = time.time()
        resp = run_single_decision(text, schema, instructions)
        elapsed = time.time() - t0

        if resp.status_code == 200:
            data = resp.json()
            result = data.get("results", [{}])[0]
            fields = result.get("fields", {})
            field_data = fields.get(expected_field, {})
            predicted = field_data.get("value", "UNKNOWN")
            probability = field_data.get("probability", 0.0)
            tokens = result.get("usage", {}).get("context_tokens", 0)
            timings = data.get("timings", {})

            expected = test["expected"]
            correct = predicted == expected
            status = "PASS" if correct else "FAIL"
            prob_str = f"{probability*100:.1f}%"

            mark = "" if correct else f" [expected: {expected}]"
            print(f"[{i+1:2d}/{len(tests)}] {test['file']:<8} {status} -> {predicted:<10} ({prob_str}) [{elapsed*1000:.0f}ms, {tokens} toks]{mark}")

            results.append({
                "file": test["file"],
                "predicted": predicted,
                "expected": expected,
                "correct": correct,
                "probability": probability,
                "elapsed_ms": round(elapsed * 1000, 1),
                "context_tokens": tokens,
                "difficulty": test["difficulty"],
                "category": test["category"],
                "timings": timings,
            })
        else:
            print(f"[{i+1:2d}/{len(tests)}] {test['file']:<8} HTTP {resp.status_code}: {resp.text[:100]}")
            results.append({
                "file": test["file"],
                "error": f"HTTP {resp.status_code}",
                "correct": False,
                "difficulty": test["difficulty"],
                "category": test["category"],
            })

    total_elapsed = time.time() - t_total

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    correct_count = sum(1 for r in results if r.get("correct"))
    total_count = len(results)
    print(f"Accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/total_count*1000:.0f}ms per test avg)")

    # By difficulty
    print("\nBy Difficulty:")
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r.get("difficulty") == diff]
        if diff_results:
            diff_correct = sum(1 for r in diff_results if r.get("correct"))
            correct_probs = [r.get("probability", 0) for r in diff_results if r.get("correct")]
            wrong_probs = [r.get("probability", 0) for r in diff_results if not r.get("correct") and "probability" in r]
            avg_prob = sum(correct_probs) / len(correct_probs) if correct_probs else 0
            avg_wrong = sum(wrong_probs) / len(wrong_probs) if wrong_probs else 0
            print(f"  {diff:<8}: {diff_correct}/{len(diff_results)} ({diff_correct/len(diff_results)*100:.0f}%)  avg_prob={avg_prob:.1%}  avg_wrong_prob={avg_wrong:.1%}")

    # By category
    print("\nBy Category:")
    categories = sorted(set(r.get("category", "") for r in results))
    for cat in categories:
        cat_results = [r for r in results if r.get("category") == cat]
        cat_correct = sum(1 for r in cat_results if r.get("correct"))
        print(f"  {cat:<14}: {cat_correct}/{len(cat_results)} ({cat_correct/len(cat_results)*100:.0f}%)")

    # Calibration check
    correct_probs = [r["probability"] for r in results if r.get("correct") and "probability" in r]
    wrong_probs = [r["probability"] for r in results if not r.get("correct") and "probability" in r]
    if correct_probs and wrong_probs:
        avg_correct = sum(correct_probs) / len(correct_probs)
        avg_wrong = sum(wrong_probs) / len(wrong_probs)
        print(f"\nCalibration:")
        print(f"  Avg confidence (correct answers): {avg_correct:.1%}")
        print(f"  Avg confidence (wrong answers):   {avg_wrong:.1%}")
        print(f"  Well calibrated: {'YES' if avg_correct > avg_wrong else 'NO'}")

    # Wrong answers
    wrong = [r for r in results if not r.get("correct")]
    if wrong:
        print(f"\nWrong answers ({len(wrong)}):")
        for r in wrong:
            print(f"  {r['file']}: expected '{r.get('expected', '?')}', got '{r.get('predicted', '?')}' ({r.get('probability', 0):.1%})")

    # Save results
    results_path = os.path.join(SAMPLES_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "question": question,
            "schema": schema,
            "total_tests": total_count,
            "passed": correct_count,
            "failed": total_count - correct_count,
            "accuracy": f"{correct_count/total_count*100:.1f}%",
            "total_time_ms": round(total_elapsed * 1000, 1),
            "avg_time_ms": round(total_elapsed / total_count * 1000, 1),
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")

    return 0 if correct_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
