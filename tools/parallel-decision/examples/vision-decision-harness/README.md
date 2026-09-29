# Vision Decision Harness

A web-based UI for testing the multimodal `/decision` endpoint of llama-server, built on top of the [parallel-decision branch](https://github.com/thecodacus/llama.cpp) of llama.cpp.

## Overview

The harness provides a browser-based interface for evaluating multimodal decision-making capabilities of llama-server. It supports:

- **Image batch processing**: Upload folders of images and run classification questions
- **Image Selection mode**: Scan a folder of images and find which one matches a question (e.g. "which image contains a rubber duck?")
- **Text file batch processing**: Run text files through the decision endpoint for scoring/classification evaluation
- **Dynamic decision tags**: Add/remove decision options with a + button (tag-list style)
- **Prompt autocompletion**: Get suggestions from the llama-server `/completion` endpoint
- **Benchmarking**: Per-file timing, total batch timing, progress bar
- **Streaming results**: Results appear as each file completes (SSE)
- **Image preview toggle**: Show/hide thumbnails in results
- **Export snippet**: Generate a Python API request snippet with docstring
- **Secret scanning**: Automated scan for sensitive information before committing

## Prerequisites

1. **llama-server with decision + multimodal support** — built from the `multimodal-decision` branch of the [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) fork. The server must be started with `--decision-seqs N` (N >= 3) and `--mmproj` for vision support.

   ```bash
   ./build/bin/llama-server \
     --host 0.0.0.0 \
     --model model-Q4_K_M.gguf \
     --mmproj mmproj-model.gguf \
     --decision-seqs 8 \
     --device Vulkan1 \
     --port 8081
   ```

2. **Python 3** with Flask and requests:
   ```bash
   pip install flask flask-cors requests
   ```

## Running

```bash
cd vision-decision-harness
python3 app.py
```

The UI is available at `http://0.0.0.0:5786`.

The server URL defaults to `http://0.0.0.0:8081`. Override with:
```bash
LLAMA_SERVER_URL=http://localhost:8081 python3 app.py
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main UI |
| `/api/health` | GET | Health check (proxies to server) |
| `/api/decision` | POST | Proxy to llama-server `/decision` |
| `/api/completion` | POST | Proxy to llama-server `/completion` |
| `/api/upload` | POST | Upload files/folders from browser |
| `/api/batch` | POST | Batch process files (non-streaming) |
| `/api/batch-stream` | POST | Batch process files with SSE streaming |
| `/api/image-selection/stream` | POST | Image selection mode with SSE streaming |
| `/api/export-snippet` | POST | Generate Python API request snippet |
| `/api/tag-suggestions` | POST | Get tag suggestions from `/completion` |

## Image Selection Mode

The image selection mode sends all images in a folder to the `/decision` endpoint in a single batched request with a yes/no schema. Each image is evaluated as a separate context, and results are aggregated to identify the single best match.

**Request:**
```json
{
  "source": "/path/to/image/folder",
  "question": "Which image contains a yellow rubber duck?",
  "instructions": "Look at this image and determine if it contains the described object. Answer YES if it does, NO if it does not.",
  "seed": 42
}
```

**SSE Events:**
- `start`: Found N images, question
- `progress`: Per-image file loaded
- `result`: Per-image yes/no evaluation
- `decision`: Single winning image with file name and probability
- `done`: Total time

## Text Test Suite

The `tests/` directory contains a comprehensive text-based decision scoring test suite:

- `text_samples/` — 20 text files with sentiment classification questions
- `answer_key.json` — Expected results for all test files
- `schema.json` — Shared JSON Schema for the tests
- `evaluate_text_tests.py` — Runner that sends each file through `/decision` and compares to the answer key
- `test_decision_scoring.py` — 41 built-in decision scoring tests (text-only)

Run with:
```bash
python3 tests/evaluate_text_tests.py
python3 tests/test_decision_scoring.py
```

## Secret Scanning

Before committing, scan all harness files and modified llama.cpp files for sensitive information:

```bash
python3 tests/scan_for_secrets.py
```

This script uses the llama-server `/decision` endpoint to classify each file as containing or not containing:
- Passwords
- API keys
- Email addresses
- Phone numbers
- Home directory paths
- IP addresses
- Credentials/tokens

Any files flagged as sensitive are reported for manual review before committing.

## Multimodal Decision Support

### Server-side Changes

The fork adds the following to `tools/server/server-context.cpp`:

1. **Multi-image contexts**: The `images` field can be an array of arrays — one sub-array per context, enabling multiple images per decision context.

2. **Media marker handling**: Context text can contain inline media markers (fetched from `/props`) to position images at specific points in the text. If no markers are present, they are prepended for backward compatibility.

3. **Batch vision encoding path**: Uses `mtmd_batch_init` → `mtmd_batch_add_chunk` → `mtmd_batch_encode` → `mtmd_batch_get_output_embd` → `mtmd_helper_decode_image_chunk` for vision encoding, matching the server's working completion path. Falls back to per-chene encoding for models that don't support batch encoding (e.g., SmolVLM2).

### Decision Engine Changes

- `decision-engine.h/cpp` — Added `multimodal_tokens` struct, `tokenize_mm()` method, `options::context_bitmaps` field, and batch vision encoding in `decode_parts()`
- `tools/mtmd/mtmd.h` — Added explicit move constructors for `bitmaps` and `bitmap` to support vector storage
- `tools/parallel-decision/CMakeLists.txt` — Links `mtmd` target

### Debug Toggle

All debug prints in the decision engine are controlled by the `LLAMA_DECISION_DEBUG` environment variable:

```bash
LLAMA_DECISION_DEBUG=1 ./build/bin/llama-server --model ...
```

## License

This harness is provided as an example for the parallel-decision multimodal extensions. See the main [llama.cpp README](https://github.com/ggml-org/llama.cpp) for the underlying project license.
