#!/usr/bin/env python3
"""
Test suite for evaluating the llama-server /decision endpoint's
scoring and classification abilities.

These are TEXT-ONLY tests — no image references. Each test presents
a text context and asks the model to classify or choose among options
based on the textual content alone.
"""

import json
import os
import base64
import time
import requests
from pathlib import Path

# ---- Configuration ----
SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://0.0.0.0:8081")
TEST_IMAGE_DIR = os.environ.get("TEST_IMAGE_DIR",
                                os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "llama.cpp-thecodacus", "tools", "mtmd"))


# ---- Test Cases (TEXT-ONLY, no image references) ----
TESTS = [
    # 1. Sentiment classification
    {
        "id": "t01_sentiment_pos",
        "context": "The product arrived early and exceeded all expectations. The packaging was perfect and the quality is outstanding.",
        "schema": {"properties": {"sentiment": {"type": "string",
            "enum": ["positive", "negative", "neutral"]}}},
        "expected": "positive",
        "difficulty": "easy",
        "category": "sentiment",
    },
    # 2. Sentiment - negative
    {
        "id": "t02_sentiment_neg",
        "context": "This service was terrible. The staff was rude, the food was cold, and I had to wait an hour. Never coming back.",
        "schema": {"properties": {"sentiment": {"type": "string",
            "enum": ["positive", "negative", "neutral"]}}},
        "expected": "negative",
        "difficulty": "easy",
        "category": "sentiment",
    },
    # 3. Sentiment - neutral
    {
        "id": "t03_sentiment_neu",
        "context": "The meeting is scheduled for 3 PM in conference room B. Please bring the quarterly report.",
        "schema": {"properties": {"sentiment": {"type": "string",
            "enum": ["positive", "negative", "neutral"]}}},
        "expected": "neutral",
        "difficulty": "medium",
        "category": "sentiment",
    },
    # 4. Urgency classification
    {
        "id": "t04_urgency_high",
        "context": "The production server is down and customers cannot complete purchases. Immediate action required.",
        "schema": {"properties": {"urgency": {"type": "string",
            "enum": ["low", "medium", "high", "critical"]}}},
        "expected": "critical",
        "difficulty": "easy",
        "category": "urgency",
    },
    # 5. Urgency - medium
    {
        "id": "t05_urgency_med",
        "context": "Please review the design documents and provide feedback by end of week.",
        "schema": {"properties": {"urgency": {"type": "string",
            "enum": ["low", "medium", "high", "critical"]}}},
        "expected": "medium",
        "difficulty": "medium",
        "category": "urgency",
    },
    # 6. Urgency - low
    {
        "id": "t06_urgency_low",
        "context": "When you have time, could you update the project wiki with the new API endpoints?",
        "schema": {"properties": {"urgency": {"type": "string",
            "enum": ["low", "medium", "high", "critical"]}}},
        "expected": "low",
        "difficulty": "hard",
        "category": "urgency",
    },
    # 7. Action selection
    {
        "id": "t07_action_ok",
        "context": "The user has confirmed their email address and completed the registration form. All validation checks passed.",
        "schema": {"properties": {"action": {"type": "string",
            "enum": ["CLICK_OK", "CLICK_CANCEL", "NO_ACTION"]}}},
        "expected": "CLICK_OK",
        "difficulty": "easy",
        "category": "action",
    },
    # 8. Action - cancel
    {
        "id": "t08_action_cancel",
        "context": "The payment was declined due to insufficient funds. The user needs to provide alternative payment.",
        "schema": {"properties": {"action": {"type": "string",
            "enum": ["CLICK_OK", "CLICK_CANCEL", "NO_ACTION"]}}},
        "expected": "CLICK_CANCEL",
        "difficulty": "medium",
        "category": "action",
    },
    # 9. Action - no action
    {
        "id": "t09_action_none",
        "context": "All system checks passed. The service is running normally with no issues detected. No further action needed.",
        "schema": {"properties": {"action": {"type": "string",
            "enum": ["CLICK_OK", "CLICK_CANCEL", "NO_ACTION"]}}},
        "expected": "NO_ACTION",
        "difficulty": "hard",
        "category": "action",
    },
    # 10. Content type classification
    {
        "id": "t10_content_news",
        "context": "Breaking: The city council voted today to approve the new budget proposal. The measure passed with a 7-2 majority after three hours of debate.",
        "schema": {"properties": {"type": {"type": "string",
            "enum": ["news", "opinion", "advertisement", "instruction", "summary"]}}},
        "expected": "news",
        "difficulty": "easy",
        "category": "content",
    },
    # 11. Content type - instruction
    {
        "id": "t11_content_instruction",
        "context": "To reset your password, first navigate to the login page. Click the 'Forgot Password' link. Enter your email address and submit.",
        "schema": {"properties": {"type": {"type": "string",
            "enum": ["news", "opinion", "advertisement", "instruction", "summary"]}}},
        "expected": "instruction",
        "difficulty": "medium",
        "category": "content",
    },
    # 12. Content type - summary
    {
        "id": "t12_content_summary",
        "context": "In summary, the experiment demonstrated a significant improvement in performance. Key findings include a 15% increase in throughput and 20% reduction in latency compared to baseline.",
        "schema": {"properties": {"type": {"type": "string",
            "enum": ["news", "opinion", "advertisement", "instruction", "summary"]}}},
        "expected": "summary",
        "difficulty": "medium",
        "category": "content",
    },
    # 13. Priority classification
    {
        "id": "t13_priority_high",
        "context": "Security vulnerability detected in production. SQL injection risk in the user authentication endpoint. Fix immediately.",
        "schema": {"properties": {"priority": {"type": "string",
            "enum": ["low", "medium", "high", "urgent"]}}},
        "expected": "urgent",
        "difficulty": "medium",
        "category": "priority",
    },
    # 14. Priority - low
    {
        "id": "t14_priority_low",
        "context": "Consider updating the style guide documentation when time permits next quarter.",
        "schema": {"properties": {"priority": {"type": "string",
            "enum": ["low", "medium", "high", "urgent"]}}},
        "expected": "low",
        "difficulty": "hard",
        "category": "priority",
    },
    # 15. File operation
    {
        "id": "t15_file_save",
        "context": "The user has finished editing the document and clicked the save button. The changes should be persisted to disk.",
        "schema": {"properties": {"operation": {"type": "string",
            "enum": ["save", "delete", "rename", "copy", "move"]}}},
        "expected": "save",
        "difficulty": "easy",
        "category": "fileop",
    },
    # 16. File operation - delete
    {
        "id": "t16_file_delete",
        "context": "The temporary cache files from last week's processing run are no longer needed and should be removed to free up space.",
        "schema": {"properties": {"operation": {"type": "string",
            "enum": ["save", "delete", "rename", "copy", "move"]}}},
        "expected": "delete",
        "difficulty": "medium",
        "category": "fileop",
    },
    # 17. File type
    {
        "id": "t17_filetype_image",
        "context": "File: photo_2024_09_25_143022.jpg — JPEG image, 1920x1080 pixels, ICC profile: sRGB, EXIF data present, camera: iPhone 14 Pro.",
        "schema": {"properties": {"filetype": {"type": "string",
            "enum": ["image", "document", "spreadsheet", "presentation", "archive"]}}},
        "expected": "image",
        "difficulty": "easy",
        "category": "filetype",
    },
    # 18. File type - document
    {
        "id": "t18_filetype_doc",
        "context": "File: quarterly_report.pdf — PDF document, 42 pages, contains tables, charts, and formatted text. Last modified: 2024-09-20.",
        "schema": {"properties": {"filetype": {"type": "string",
            "enum": ["image", "document", "spreadsheet", "presentation", "archive"]}}},
        "expected": "document",
        "difficulty": "medium",
        "category": "filetype",
    },
    # 19. Error type
    {
        "id": "t19_error_notfound",
        "context": "Error: Resource not found at /api/users/12345. The requested user ID does not exist in the database.",
        "schema": {"properties": {"error": {"type": "string",
            "enum": ["syntax", "runtime", "network", "permission", "not_found"]}}},
        "expected": "not_found",
        "difficulty": "hard",
        "category": "error",
    },
    # 20. Error type - permission
    {
        "id": "t20_error_permission",
        "context": "Access denied: User does not have permission to read /etc/shadow. Required role: root, current role: standard_user.",
        "schema": {"properties": {"error": {"type": "string",
            "enum": ["syntax", "runtime", "network", "permission", "not_found"]}}},
        "expected": "permission",
        "difficulty": "hard",
        "category": "error",
    },
    # 21. UI element type
    {
        "id": "t21_ui_button",
        "context": "The form contains a blue rectangular button labeled 'Submit Order'. Clicking it sends the form data to the server.",
        "schema": {"properties": {"element": {"type": "string",
            "enum": ["button", "link", "checkbox", "text_field", "dropdown"]}}},
        "expected": "button",
        "difficulty": "easy",
        "category": "ui",
    },
    # 22. UI element - dropdown
    {
        "id": "t22_ui_dropdown",
        "context": "The settings panel shows a dropdown menu with options: Light Mode, Dark Mode, Auto. The current selection is Dark Mode.",
        "schema": {"properties": {"element": {"type": "string",
            "enum": ["button", "link", "checkbox", "text_field", "dropdown"]}}},
        "expected": "dropdown",
        "difficulty": "medium",
        "category": "ui",
    },
    # 23. Language detection
    {
        "id": "t23_lang_english",
        "context": "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump!",
        "schema": {"properties": {"lang": {"type": "string",
            "enum": ["english", "spanish", "french", "german", "chinese"]}}},
        "expected": "english",
        "difficulty": "easy",
        "category": "language",
    },
    # 24. Language - Spanish
    {
        "id": "t24_lang_spanish",
        "context": "Hola, como estas? Me encantaria visitar Espana algun dia. La comida espanola es deliciosa, especialmente la paella.",
        "schema": {"properties": {"lang": {"type": "string",
            "enum": ["english", "spanish", "french", "german", "chinese"]}}},
        "expected": "spanish",
        "difficulty": "medium",
        "category": "language",
    },
    # 25. Navigation
    {
        "id": "t25_navigation_settings",
        "context": "To change your notification preferences, go to the account section and look for the bell icon. Click it to open the settings panel.",
        "schema": {"properties": {"destination": {"type": "string",
            "enum": ["home", "settings", "profile", "logout", "help"]}}},
        "expected": "settings",
        "difficulty": "hard",
        "category": "navigation",
    },
    # 26. Format detection
    {
        "id": "t26_format_json",
        "context": '{"users": [{"name": "Alice", "id": 1}, {"name": "Bob", "id": 2}], "count": 2}',
        "schema": {"properties": {"format": {"type": "string",
            "enum": ["plain", "markdown", "html", "json", "xml"]}}},
        "expected": "json",
        "difficulty": "easy",
        "category": "format",
    },
    # 27. Format - markdown
    {
        "id": "t27_format_markdown",
        "context": "# Project README\n\nThis project does **important things**. See the [documentation](docs.md) for details.\n\n```python\nprint('hello')\n```",
        "schema": {"properties": {"format": {"type": "string",
            "enum": ["plain", "markdown", "html", "json", "xml"]}}},
        "expected": "markdown",
        "difficulty": "medium",
        "category": "format",
    },
    # 28. Interaction type
    {
        "id": "t28_interaction_click",
        "context": "The user pressed the red submit button with their mouse. The button highlighted blue briefly and then the form was submitted.",
        "schema": {"properties": {"interaction": {"type": "string",
            "enum": ["click", "hover", "drag", "scroll", "type"]}}},
        "expected": "click",
        "difficulty": "easy",
        "category": "interaction",
    },
    # 29. Interaction - scroll
    {
        "id": "t29_interaction_scroll",
        "context": "The user moved the scrollbar down using the mouse wheel to see more content below the fold of the webpage.",
        "schema": {"properties": {"interaction": {"type": "string",
            "enum": ["click", "hover", "drag", "scroll", "type"]}}},
        "expected": "scroll",
        "difficulty": "hard",
        "category": "interaction",
    },
    # 30. State classification
    {
        "id": "t30_state_on",
        "context": "The switch is in the active position. The LED indicator is lit green. Power is flowing to the connected device.",
        "schema": {"properties": {"state": {"type": "string",
            "enum": ["on", "off", "indeterminate", "disabled"]}}},
        "expected": "on",
        "difficulty": "easy",
        "category": "state",
    },
    # 31. State - off
    {
        "id": "t31_state_off",
        "context": "The device is powered down. The LED is unlit. No power is being drawn from the battery. The switch is in the inactive position.",
        "schema": {"properties": {"state": {"type": "string",
            "enum": ["on", "off", "indeterminate", "disabled"]}}},
        "expected": "off",
        "difficulty": "medium",
        "category": "state",
    },
    # 32. Data source
    {
        "id": "t32_data_api",
        "context": "The frontend fetches user data by making a GET request to /api/v2/users with authentication headers. Results are returned as JSON.",
        "schema": {"properties": {"source": {"type": "string",
            "enum": ["database", "api", "file", "cache", "user_input"]}}},
        "expected": "api",
        "difficulty": "medium",
        "category": "data",
    },
    # 33. Data source - user_input
    {
        "id": "t33_data_user",
        "context": "The search query was typed directly into the search box by the user. No predefined data source was queried.",
        "schema": {"properties": {"source": {"type": "string",
            "enum": ["database", "api", "file", "cache", "user_input"]}}},
        "expected": "user_input",
        "difficulty": "hard",
        "category": "data",
    },
    # 34. Security level
    {
        "id": "t34_security_internal",
        "context": "This document contains company-wide policies for internal use only. It is accessible to all employees but not to external parties.",
        "schema": {"properties": {"level": {"type": "string",
            "enum": ["public", "internal", "confidential", "restricted"]}}},
        "expected": "internal",
        "difficulty": "medium",
        "category": "security",
    },
    # 35. Security - public
    {
        "id": "t35_security_public",
        "context": "The marketing brochure is available on the company website for anyone to download and share freely.",
        "schema": {"properties": {"level": {"type": "string",
            "enum": ["public", "internal", "confidential", "restricted"]}}},
        "expected": "public",
        "difficulty": "easy",
        "category": "security",
    },
    # 36. Time urgency
    {
        "id": "t36_time_soon",
        "context": "Please review the draft proposal and provide feedback within the next few days. The deadline is Friday.",
        "schema": {"properties": {"urgency": {"type": "string",
            "enum": ["immediate", "soon", "later", "anytime"]}}},
        "expected": "soon",
        "difficulty": "hard",
        "category": "time",
    },
    # 37. Time urgency - later
    {
        "id": "t37_time_later",
        "context": "When you have time over the next few weeks, please update the project documentation with the new API changes.",
        "schema": {"properties": {"urgency": {"type": "string",
            "enum": ["immediate", "soon", "later", "anytime"]}}},
        "expected": "later",
        "difficulty": "hard",
        "category": "time",
    },
    # 38. Direction
    {
        "id": "t38_direction_right",
        "context": "The carousel should advance to the next item. The user clicked the right arrow button to move forward.",
        "schema": {"properties": {"direction": {"type": "string",
            "enum": ["left", "right", "up", "down", "none"]}}},
        "expected": "right",
        "difficulty": "medium",
        "category": "direction",
    },
    # 39. Direction - left
    {
        "id": "t39_direction_left",
        "context": "The user wants to go back to the previous slide. Click the left arrow to navigate backwards.",
        "schema": {"properties": {"direction": {"type": "string",
            "enum": ["left", "right", "up", "down", "none"]}}},
        "expected": "left",
        "difficulty": "hard",
        "category": "direction",
    },
    # 40. Component type
    {
        "id": "t40_component_card",
        "context": "Each user profile is displayed in a bordered box with a shadow. It contains the profile picture, name, and status below.",
        "schema": {"properties": {"component": {"type": "string",
            "enum": ["modal", "sidebar", "header", "footer", "card"]}}},
        "expected": "card",
        "difficulty": "medium",
        "category": "ui",
    },
    # 41. Component - sidebar
    {
        "id": "t41_component_sidebar",
        "context": "The navigation panel on the left side of the screen contains links to Dashboard, Settings, and Logout.",
        "schema": {"properties": {"component": {"type": "string",
            "enum": ["modal", "sidebar", "header", "footer", "card"]}}},
        "expected": "sidebar",
        "difficulty": "hard",
        "category": "ui",
    },
]


def run_single_decision(test_case, timeout=120):
    """Run a single decision test and return the result dict."""
    body = {
        "contexts": [test_case["context"]],
        "schema": test_case["schema"],
        "instructions": "Select the correct value from the allowed choices based on the context provided.",
        "seed": 42,
    }

    t0 = time.time()
    try:
        resp = requests.post(f"{SERVER_URL}/decision", json=body, timeout=timeout)
        elapsed = time.time() - t0
        if resp.status_code == 200:
            data = resp.json()
            field_name = list(data["results"][0]["fields"].keys())[0]
            fld = data["results"][0]["fields"][field_name]
            return {
                "success": True,
                "decision": fld["value"],
                "probability": fld["probability"],
                "expected": test_case["expected"],
                "elapsed_ms": round(elapsed * 1000),
                "tokens": data["results"][0]["usage"].get("context_tokens", 0),
                "timings": data.get("timings", {}),
                "id": test_case["id"],
                "context": test_case["context"],
                "difficulty": test_case["difficulty"],
                "category": test_case["category"],
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "id": test_case["id"],
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "id": test_case["id"],
        }


def evaluate_results(results):
    """Evaluate test results and print a summary."""
    passed = 0
    failed = 0
    total_prob_correct = 0.0
    total_prob_wrong = 0.0
    correct_count = 0
    wrong_count = 0

    print("\n" + "=" * 90)
    print("DECISION SCORING TEST RESULTS (TEXT-ONLY)")
    print("=" * 90)
    print(f"{'ID':<18} {'Diff':<8} {'Category':<14} {'Expected':<14} {'Got':<14} {'Prob':<8} {'Time':<8} {'Status':<6}")
    print("-" * 90)

    by_difficulty = {}
    by_category = {}

    for r in results:
        if not r["success"]:
            print(f"{r['id']:<18} {'ERROR':<8} {'':<14} {'N/A':<14} {'N/A':<14} {'N/A':<8} {'N/A':<8} FAIL")
            failed += 1
            continue

        is_correct = r["decision"] == r["expected"]
        status = "PASS" if is_correct else "FAIL"
        if is_correct:
            passed += 1
            total_prob_correct += r["probability"]
            correct_count += 1
        else:
            failed += 1
            total_prob_wrong += r["probability"]
            wrong_count += 1

        diff = r["difficulty"]
        if diff not in by_difficulty:
            by_difficulty[diff] = {"passed": 0, "total": 0, "probs": []}
        by_difficulty[diff]["total"] += 1
        if is_correct:
            by_difficulty[diff]["passed"] += 1
            by_difficulty[diff]["probs"].append(r["probability"])

        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"passed": 0, "total": 0, "probs": []}
        by_category[cat]["total"] += 1
        if is_correct:
            by_category[cat]["passed"] += 1
            by_category[cat]["probs"].append(r["probability"])

        print(f"{r['id']:<18} {diff:<8} {cat:<14} {r['expected']:<14} {r['decision']:<14} {r['probability']:.2%} {str(r['elapsed_ms'])+'ms':<8} {status}")

    print("-" * 90)
    print(f"\nOverall: {passed}/{len(results)} passed ({passed/len(results)*100:.1f}%)\n")

    print("By Difficulty:")
    for diff in sorted(by_difficulty.keys()):
        d = by_difficulty[diff]
        avg_prob = sum(d["probs"]) / len(d["probs"]) if d["probs"] else 0
        print(f"  {diff:<8}: {d['passed']}/{d['total']} ({d['passed']/d['total']*100:.1f}%)  avg_prob={avg_prob:.2%}")

    print("\nBy Category:")
    for cat in sorted(by_category.keys()):
        c = by_category[cat]
        avg_prob = sum(c["probs"]) / len(c["probs"]) if c["probs"] else 0
        print(f"  {cat:<14}: {c['passed']}/{c['total']} ({c['passed']/c['total']*100:.1f}%)  avg_prob={avg_prob:.2%}")

    # Calibration analysis
    if correct_count > 0 and wrong_count > 0:
        avg_correct = total_prob_correct / correct_count
        avg_wrong = total_prob_wrong / wrong_count
        print(f"\nCalibration:")
        print(f"  Avg prob (correct):   {avg_correct:.2%}")
        print(f"  Avg prob (wrong):     {avg_wrong:.2%}")
        if avg_correct > avg_wrong:
            print(f"  -> WELL CALIBRATED (correct answers have higher confidence)")
        else:
            print(f"  -> POORLY CALIBRATED (wrong answers have higher confidence)")
    elif correct_count > 0:
        print(f"\nCalibration: All answers correct ({total_prob_correct/correct_count:.2%} avg prob)")

    # Timing summary
    valid_times = [r["elapsed_ms"] for r in results if r["success"]]
    if valid_times:
        print(f"\nTiming:")
        print(f"  Avg per decision: {sum(valid_times)/len(valid_times):.0f}ms")
        print(f"  Min: {min(valid_times)}ms | Max: {max(valid_times)}ms")
        total_time = sum(valid_times)
        print(f"  Total: {total_time}ms ({total_time/1000:.1f}s)")

    return passed, failed


def save_answer_key():
    """Save the answer key separately from the test code."""
    key = {
        "description": "Answer key for decision scoring test suite (text-only)",
        "total_tests": len(TESTS),
        "tests": [
            {
                "id": t["id"],
                "context": t["context"][:80] + "..." if len(t["context"]) > 80 else t["context"],
                "expected": t["expected"],
                "difficulty": t["difficulty"],
                "category": t["category"],
                "schema_enum": list(t["schema"]["properties"].values())[0]["enum"],
            }
            for t in TESTS
        ],
    }

    key_path = os.path.join(os.path.dirname(__file__), "answer_key.json")
    with open(key_path, "w") as f:
        json.dump(key, f, indent=2)
    print(f"Answer key saved to {key_path}")
    return key_path


if __name__ == "__main__":
    import sys

    save_key = "--save-key" in sys.argv

    if save_key:
        save_answer_key()
        print("Answer key saved. Run without --save-key to test.")
        sys.exit(0)

    print(f"Running {len(TESTS)} decision scoring tests (text-only)...")
    print(f"Server: {SERVER_URL}")

    results = []
    for i, test in enumerate(TESTS):
        r = run_single_decision(test)
        results.append(r)
        if r["success"]:
            status = "PASS" if r["decision"] == r["expected"] else "FAIL"
            print(f"[{i+1:2d}/{len(TESTS)}] {test['id']}: {status} -> {r['decision']} ({r['probability']:.1%}) [{r['elapsed_ms']}ms]")
        else:
            print(f"[{i+1:2d}/{len(TESTS)}] {test['id']}: ERROR: {r.get('error', 'unknown')}")

    passed, failed = evaluate_results(results)

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")

    sys.exit(0 if failed == 0 else 1)
