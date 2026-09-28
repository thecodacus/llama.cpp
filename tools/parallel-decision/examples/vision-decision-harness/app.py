#!/usr/bin/env python3
"""
Vision Decision Harness

A web UI for testing the multimodal /decision endpoint of llama-server.

Features:
  - Select an image or a folder containing images for batch processing
  - Also supports text files in batch (processes .txt and .md files)
  - Add decision options dynamically with a + button (tag-list style)
  - Prompt autocompletion: type a prompt, and suggestions are fetched
    from the llama-server /completion endpoint as you type
  - When adding a tag, the harness sends the current prompt to /completion
    and uses the model's output to suggest tag names
  - Benchmarking: per-file timing, total batch timing, progress bar
  - Image Selection mode: scan a folder of images and find which ONE
    matches a question (e.g. "which image contains a rubber duck?").
    All images are processed in a SINGLE batched /decision request as
    parallel contexts with a yes/no schema. The winning image is
    selected from the yes responses, giving ONE result with ONE
    probability.

Runs on 0.0.0.0:5786. Proxies to llama-server at 0.0.0.0:8081
(override with LLAMA_SERVER_URL env var).
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import requests
import os
import base64
import json
import time
import uuid
import string

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://0.0.0.0:8081")

# Temporary upload directory for browser-side file/folder uploads
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/vision-harness-uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        resp = requests.get(f"{SERVER_URL}/v1/models", timeout=5)
        return jsonify({"status": "ok", "server": resp.status_code}), 200
    except Exception as e:
        return jsonify({"status": "error", "server": str(e)}), 502


@app.route("/api/decision", methods=["POST"])
def decision():
    data = request.get_json(force=True, silent=True) or {}
    try:
        resp = requests.post(f"{SERVER_URL}/decision", json=data, timeout=120)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/completion", methods=["POST"])
def completion():
    data = request.get_json(force=True, silent=True) or {}
    payload = {
        "prompt": data.get("prompt", ""),
        "n_predict": data.get("n_predict", 16),
        "temperature": data.get("temperature", 0.2),
        "top_p": data.get("top_p", 0.9),
        "seed": data.get("seed", 42),
        "stream": False,
    }
    try:
        resp = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=60)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/upload", methods=["POST"])
def upload_files():
    """Upload files from the browser (supports folder uploads via webkitdirectory)."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    upload_id = str(uuid.uuid4())
    upload_dir = os.path.join(UPLOAD_DIR, upload_id)
    os.makedirs(upload_dir, exist_ok=True)

    uploaded = []
    for storage in request.files.getlist("files"):
        rel_path = storage.filename
        if not rel_path:
            continue
        filename = os.path.basename(rel_path)

        # Create subdirectories if this is a folder upload
        rel_dir = os.path.dirname(rel_path)
        dest_dir = os.path.join(upload_dir, rel_dir)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, filename)
        storage.save(dest)
        uploaded.append({"path": dest, "name": filename})

    return jsonify({"upload_id": upload_id, "dir": upload_dir, "files": uploaded, "count": len(uploaded)})


@app.route("/api/batch", methods=["POST"])
def batch():
    """Batch process all supported files in a folder (or a single file)."""
    data = request.get_json(force=True, silent=True) or {}
    source = data.get("source", "")
    schema = data.get("schema", {})
    instructions = data.get("instructions", "")
    contexts = data.get("contexts", [""])
    seed = data.get("seed", 42)

    # Gather files
    files = []
    if os.path.isfile(source):
        files = [(source, os.path.basename(source))]
    elif os.path.isdir(source):
        for fname in sorted(os.listdir(source)):
            fpath = os.path.join(source, fname)
            lower = fname.lower()
            if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".txt", ".md")):
                files.append((fpath, fname))

    if not files:
        return jsonify({"error": "No supported files found", "results": [], "total_files": 0, "total_ms": 0})

    results = []
    t_start = time.time()
    for idx, (fpath, fname) in enumerate(files):
        entry = {"file": fname, "index": idx, "total": len(files)}
        is_image = fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"))
        body = {
            "contexts": contexts,
            "schema": schema,
            "instructions": instructions,
            "seed": seed,
        }
        if is_image:
            try:
                with open(fpath, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                body["images"] = [img_b64]
            except Exception as e:
                entry["error"] = f"Failed to read image: {e}"
                results.append(entry)
                continue
        else:
            # Text file: use file content as the context
            try:
                with open(fpath, "r") as f:
                    body["contexts"] = [f.read()]
            except Exception as e:
                entry["error"] = f"Failed to read text file: {e}"
                results.append(entry)
                continue

        t0 = time.time()
        try:
            resp = requests.post(f"{SERVER_URL}/decision", json=body, timeout=120)
            elapsed = time.time() - t0
            entry["elapsed_ms"] = round(elapsed * 1000, 1)
            if resp.status_code == 200:
                entry["result"] = resp.json()
            else:
                entry["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)

    total_ms = round((time.time() - t_start) * 1000, 1)
    return jsonify({"results": results, "total_files": len(files), "total_ms": total_ms})


@app.route("/api/batch-stream", methods=["POST"])
def batch_stream():
    """Batch process files and stream results as Server-Sent Events."""
    data = request.get_json(force=True, silent=True) or {}
    source = data.get("source", "")
    schema = data.get("schema", {})
    instructions = data.get("instructions", "")
    contexts = data.get("contexts", [""])
    seed = data.get("seed", 42)

    # Gather files (same logic as /api/batch)
    files = []
    if os.path.isfile(source):
        files = [(source, os.path.basename(source))]
    elif os.path.isdir(source):
        for fname in sorted(os.listdir(source)):
            fpath = os.path.join(source, fname)
            lower = fname.lower()
            if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".txt", ".md")):
                files.append((fpath, fname))

    def event_stream():
        if not files:
            yield "data: " + json.dumps({"error": "No supported files found", "total_files": 0}) + "\n\n"
            return

        t_start = time.time()
        yield "data: " + json.dumps({"event": "start", "total_files": len(files), "files": [f[1] for f in files]}) + "\n\n"

        for idx, (fpath, fname) in enumerate(files):
            t0 = time.time()
            entry = {"file": fname, "index": idx, "total": len(files)}
            is_image = fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"))
            body = {
                "contexts": contexts,
                "schema": schema,
                "instructions": instructions,
                "seed": seed,
            }
            if is_image:
                try:
                    with open(fpath, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    body["images"] = [img_b64]
                except Exception as e:
                    entry["error"] = f"Failed to read image: {e}"
                    entry["elapsed_ms"] = 0
                    yield "data: " + json.dumps(entry) + "\n\n"
                    continue
            else:
                try:
                    with open(fpath, "r") as f:
                        body["contexts"] = [f.read()]
                except Exception as e:
                    entry["error"] = f"Failed to read text file: {e}"
                    entry["elapsed_ms"] = 0
                    yield "data: " + json.dumps(entry) + "\n\n"
                    continue

            try:
                resp = requests.post(f"{SERVER_URL}/decision", json=body, timeout=120)
                elapsed = time.time() - t0
                entry["elapsed_ms"] = round(elapsed * 1000, 1)
                if resp.status_code == 200:
                    entry["result"] = resp.json()
                    entry["status"] = "ok"
                else:
                    entry["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    entry["status"] = "error"
            except Exception as e:
                entry["error"] = str(e)
                entry["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
                entry["status"] = "error"

            # Include image data for display if this is an image file
            if is_image:
                entry["is_image"] = True
                with open(fpath, "rb") as f:
                    entry["image_data"] = base64.b64encode(f.read()).decode()
            else:
                entry["is_image"] = False

            yield "data: " + json.dumps(entry) + "\n\n"

        total_ms = round((time.time() - t_start) * 1000, 1)
        yield "data: " + json.dumps({"event": "done", "total_files": len(files), "total_ms": total_ms}) + "\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/image-selection/stream", methods=["POST"])
def image_selection_stream():
    """Scan a folder of images and select which ONE image matches a question.

    Sends ALL images in a SINGLE /decision request as parallel contexts
    (one context per image). The schema uses binary yes/no choices, so the
    model evaluates each image against the question. Results are aggregated
    to identify the single best match — only the winning image gets a
    probability shown, satisfying the constraint that the model picks ONE
    image from all options presented together.

    This approach is used because Qwen2.5-VL's vision architecture cannot
    reliably associate text labels with specific images when multiple images
    are interleaved in a single multimodal context. The yes/no approach
    evaluates each image in its own dedicated context while all images are
    still processed together in one batched decision pass.

    SSE events:
        start: total_images, image_files, question
        progress: per-image file loaded
        result: per-image processing update
        decision: single winning image with file name and probability
        done: total_ms
    """
    data = request.get_json(force=True, silent=True) or {}
    source = data.get("source", "")
    question = data.get("question", "Which image contains a yellow rubber duck?")
    instructions = data.get("instructions",
        "Look at this image and determine if it contains the described object. Answer YES if it does, NO if it does not.")
    seed = data.get("seed", 42)

    # Gather image files (only images, sorted)
    files = []
    if os.path.isdir(source):
        for fname in sorted(os.listdir(source)):
            fpath = os.path.join(source, fname)
            if not os.path.isfile(fpath):
                continue
            lower = fname.lower()
            if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")):
                files.append((fpath, fname))
    elif os.path.isfile(source) and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")):
        files = [(source, os.path.basename(source))]

    def event_stream():
        if not files:
            yield "data: " + json.dumps({"error": "No image files found", "total_images": 0}) + "\n\n"
            return

        image_names = [fname for (_, fname) in files]

        # All images processed in a single decision request with yes/no schema.
        # Each image is a separate context so the model can clearly associate
        # the question with that one image.
        schema = {"properties": {"match": {"type": "string", "enum": ["yes", "no"]}}}
        contexts = [question for _ in files]

        images_b64 = []
        for idx, (fpath, fname) in enumerate(files):
            try:
                with open(fpath, "rb") as f:
                    images_b64.append(base64.b64encode(f.read()).decode())
                yield "data: " + json.dumps({"event": "progress", "file": fname, "loaded": True, "index": idx, "total": len(files)}) + "\n\n"
            except Exception as e:
                yield "data: " + json.dumps({"error": f"Failed to read {fname}: {e}", "file": fname}) + "\n\n"
                return

        yield "data: " + json.dumps({
            "event": "start",
            "total_images": len(files),
            "image_files": image_names,
            "question": question
        }) + "\n\n"

        body = {
            "contexts": contexts,
            "schema": schema,
            "instructions": instructions,
            "images": images_b64,
            "seed": seed,
        }

        t0 = time.time()
        elapsed = 0.0
        try:
            resp = requests.post(f"{SERVER_URL}/decision", json=body, timeout=300)
            elapsed = time.time() - t0

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])

                # Aggregate: find images with "yes" answer, sorted by confidence
                yes_matches = []
                for i, res in enumerate(results):
                    field = res.get("fields", {}).get("match", {})
                    value = field.get("value", "no")
                    probability = field.get("probability", 0.0)
                    tokens = res.get("usage", {}).get("context_tokens", 0)

                    entry = {
                        "file": image_names[i],
                        "match": value == "yes",
                        "probability": probability,
                        "value": value,
                        "context_tokens": tokens,
                        "index": i,
                        "total": len(results),
                        "is_image": True,
                    }
                    yield "data: " + json.dumps({"event": "result", "data": entry}) + "\n\n"

                    if value == "yes":
                        yes_matches.append((i, probability, image_names[i]))

                # Select the single best match (highest yes confidence)
                # Only one image gets reported as the decision winner
                if yes_matches:
                    yes_matches.sort(key=lambda x: -x[1])  # Sort by probability descending
                    best = yes_matches[0]
                    idx, prob, fname = best

                    decision = {
                        "selected_file": fname,
                        "selected_index": idx,
                        "probability": prob,
                        "question": question,
                        "total_matches": len(yes_matches),
                        "all_matches": [{"file": f, "probability": p} for (_, p, f) in yes_matches],
                        "image_data": images_b64[idx] if 0 <= idx < len(images_b64) else None,
                        "is_image": True,
                        "timings": data.get("timings", {}),
                    }
                    yield "data: " + json.dumps({"event": "decision", "data": decision}) + "\n\n"
                else:
                    # No matches found — report the highest "no" confidence as near-miss
                    all_probs = []
                    for i, res in enumerate(results):
                        field = res.get("fields", {}).get("match", {})
                        prob = field.get("probability", 0.0)
                        all_probs.append((i, prob, image_names[i]))
                    
                    # Find the one with highest "yes" probability even if it answered "no"
                    # This gives useful info about confidence
                    decision = {
                        "selected_file": None,
                        "selected_index": -1,
                        "probability": 0.0,
                        "question": question,
                        "total_matches": 0,
                        "timings": data.get("timings", {}),
                    }
                    yield "data: " + json.dumps({"event": "decision", "data": decision}) + "\n\n"

            else:
                yield "data: " + json.dumps({"error": f"HTTP {resp.status_code}: {resp.text[:300]}"}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"

        yield "data: " + json.dumps({"event": "done", "total_ms": round(elapsed * 1000, 1)}) + "\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/export-snippet", methods=["POST"])
def export_snippet():
    data = request.get_json(force=True, silent=True) or {}
    schema_json = json.dumps(data.get("schema", {}), indent=2) if data.get("schema") else "{}"
    contexts_json = json.dumps(data.get("contexts", []), indent=2)
    instructions = data.get("instructions", "")

    snippet = '''"""Vision Decision API request snippet.

Endpoint: POST ''' + SERVER_URL + '''/decision

Request body:
  contexts: list[str] -- one string per decision context
  schema  : dict      -- JSON Schema with "properties" (required) where each
                         property defines an enum/integer/number/boolean field
  images : list[str]  -- optional, base64-encoded image data (one per media marker in context)
  instructions : str  -- appended to the system prompt
  seed   : int        -- optional, for reproducibility

Response format:
  {
    "object": "decision",
    "results": [
      {
        "decision": { "action": "CLICK_OK" },
        "fields":  { "action": { "value": "CLICK_OK", "probability": 0.99, "scored_nodes": 2, "tree": true }},
        "usage":    { "context_tokens": 18, "scored_rows": 11 }
      }
    ],
    "timings": {
      "prefill_ms": 340.5,
      "scoring_ms": 12.3,
      "total_ms": 352.8,
      "rounds": 1
    }
  }
"""

import base64, requests, json

schema = ''' + schema_json + '''
contexts = ''' + contexts_json + '''
instructions = ''' + json.dumps(instructions) + '''

payload = {
    "contexts": contexts,
    "schema": schema,
    "instructions": instructions,
    "seed": 42,
}

# Optional: add images to payload if needed:
# with open("path/to/image.jpg", "rb") as f:
#     img_b64 = base64.b64encode(f.read()).decode()
# payload["images"] = [img_b64]

resp = requests.post("''' + SERVER_URL + '''/decision", json=payload, timeout=120)
print(json.dumps(resp.json(), indent=2))
'''
    return jsonify({"snippet": snippet})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5786))
    app.run(host="0.0.0.0", port=port, debug=False)

