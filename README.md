# OCR / PDF Extraction Benchmark

This project compares text extraction from PDFs and images across local OCR tools and cloud vision LLMs. It scores each one on text, number, table, layout and speed results.

- **Local engines:** PaddleOCR, Surya OCR, docTR, Tesseract, EasyOCR, pdfplumber, Camelot, Docling, Marker
- **Cloud vision LLMs:** OpenAI, Anthropic (Claude), DeepSeek

Each engine has its own file (for example `openai_engine.py` or `paddleocr_engine.py`), so you can run any of them alone. The Streamlit app (`app.py`) lets you upload a PDF or image, paste the correct text, and run any set of engines side by side to see which one extracts text most accurately.

Local engines use a GPU when one is available and fall back to the CPU otherwise. The cloud engines run on the provider's servers.

---

## 1. Set up the virtual environment (with `uv`)

```powershell
# From the project directory: creates .venv (Python 3.11) and installs everything
uv sync
.venv\Scripts\activate
```

`pyproject.toml` pins the GPU builds, so `uv sync` and `uv run` always install them:

| Package            | Build                    | Index                                  |
|--------------------|--------------------------|----------------------------------------|
| `torch`, `torchvision` | CUDA 12.8 (`+cu128`) | `download.pytorch.org/whl/cu128`       |
| `paddlepaddle-gpu` | 3.4.0, CUDA 12.9         | `paddlepaddle.org.cn/packages/stable/cu129` |

Both include kernels for RTX 50-series (Blackwell, sm_120) GPUs, and both fall back to the CPU on machines without an NVIDIA GPU.

> **Don't** install packages with `uv pip install torch` or `uv pip install -r requirements.txt`. PyPI's Windows `torch` wheel is CPU-only and silently replaces the CUDA build, which makes every engine run on the CPU. If that happens, `uv sync` restores the GPU builds, and the app's sidebar warns you about it.
>
> Never install the CPU `paddlepaddle` package next to `paddlepaddle-gpu`, because both use the `paddle` namespace.

### System dependencies (not installable via pip)
These are only needed by the local engines:
- **Tesseract:** https://github.com/UB-Mannheim/tesseract/wiki (Windows)
- **Ghostscript** (needed by Camelot): https://ghostscript.com/releases/gsdnld.html
- **llama.cpp `llama-server` binary** (needed by Surya OCR 0.22+ and Marker 2.0+, which run their recognition model in it). One command fetches the latest Windows build:
  ```powershell
  python scripts\install_llama_cpp.py
  ```
  On a machine with an NVIDIA GPU it installs the **CUDA** build into `.tools/llama-cpp/cuda/`; otherwise it installs the CPU build. Use `--cpu` or `--cuda` to force one. The engines always prefer the CUDA build when both are present. The sidebar shows which build is in use and offers an install button if the right one is missing.

---

## 2. API keys for the cloud engines

Copy `.env.example` to `.env` and fill in the keys you have:

```ini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
```

`.env` is git-ignored. Each cloud engine reads it when it runs, so you don't need to restart Streamlit after you edit it. If a key is missing, that engine shows an error and the other engines still run.

### Default models (flagship tier)

| Engine     | File                  | Default model     | Override with     |
|------------|-----------------------|-------------------|-------------------|
| OpenAI     | `openai_engine.py`    | `gpt-6-astra`     | `OPENAI_MODEL`    |
| Anthropic  | `anthropic_engine.py` | `claude-opus-5-5` | `ANTHROPIC_MODEL` |
| DeepSeek   | `deepseek_engine.py`  | `deepseek-flash`  | `DEEPSEEK_MODEL`  |

`deepseek-flash` is DeepSeek's only API model that accepts images, and `deepseek-v4-pro` is text-only. To compare cheaper tiers, set for example `OPENAI_MODEL=gpt-6-luna` or `ANTHROPIC_MODEL=claude-sonnet-5-5`.

DeepSeek's thinking mode is **off** by default (`DEEPSEEK_THINKING=disabled`). With it on, `deepseek-flash` spent the whole 16,000-token limit on hidden reasoning for a dense financial page and returned no text. With it off, it transcribed the same page in about 7 seconds. If any model uses up the token limit without returning text, the engine reports an error rather than a 0% "success".

Perplexity isn't included. Its Sonar chat API was retired in September 2026, and the replacement Agent API has no vision model of its own: it forwards image requests to OpenAI or Anthropic models, which this project already tests directly.

### Optional tuning

| Variable                 | Default | Meaning                                    |
|--------------------------|---------|--------------------------------------------|
| `LLM_OCR_CONCURRENCY`    | `4`     | Pages sent in parallel per engine          |
| `LLM_OCR_MAX_IMAGE_SIDE` | `2048`  | Longest side (px) of each page image sent  |
| `LLM_OCR_MAX_TOKENS`     | `16000` | Max output tokens per page                 |

---

## 3. Run the Streamlit app

```powershell
streamlit run app.py
```

1. Tick engines in the sidebar, for example **OpenAI**, **Anthropic** and **DeepSeek**.
2. Upload a PDF or image.
3. Paste the **ground-truth text** if you want accuracy scores. Without it, you only get the extracted text and timings.
4. Optionally limit the run to the current page or a page range. This keeps API costs down on long PDFs.
5. Click **Run selected engines**. The **Overall comparison** table and charts rank the engines, and you can download the results as a CSV file.

---

## 4. Run any engine by itself

Each engine runs directly on a single file:

```powershell
python openai_engine.py data\sbi_one_page.pdf
python anthropic_engine.py data\sbi_one_page.pdf
python deepseek_engine.py data\sbi_one_page.pdf

python paddleocr_engine.py sample.pdf
python suryaocr_engine.py sample.png
python doctr_engine.py sample.pdf
python tesseract_engine.py sample.jpg
python easyocr_engine.py sample.png
python pdfplumber_engine.py sample.pdf
python camelot_engine.py sample.pdf
python docling_engine.py sample.pdf
python marker_engine.py sample.pdf
```

Each script prints the first 500 characters of the extracted text, then the time taken and the device used. The cloud engines also print the model name and the input and output token counts, so you can estimate cost.

---

## 5. How the cloud engines are evaluated fairly

- **Same input for every provider.** Every page, including pages of digital PDFs, is rasterised at 200 DPI and sent as an image. This scales it to at most 2048 px on the long side and stays under each provider's size limit. PDFs are never uploaded as files, because a provider could then read the embedded text layer instead of doing OCR.
- **Same prompt.** All three engines use the same transcription prompt from `engines/vision_llm.py`. It asks for verbatim text in reading order, tables as Markdown, and `[illegible]` instead of guesses.
- **Scoring.** Before scoring, Markdown table syntax is converted to plain text so the output can be compared with a plain-text ground truth. The scores come from `engines/metrics.py`, the same code used for the local engines:
  - **Char accuracy:** character-level similarity of the normalised text
  - **Word accuracy:** word-sequence similarity
  - **Number precision / recall / F1:** whether every number in the document was extracted correctly. This is often the most important score for financial documents.
- **Tables.** Markdown tables in each model's output are parsed and shown in the **Tables** tab.
- **Layout.** LLMs don't return bounding boxes, so the **Layout (boxes)** tab is empty for these engines.
- **Speed.** The time shown is wall-clock time for all pages, including network delay, with up to `LLM_OCR_CONCURRENCY` pages sent in parallel.

The pages-to-send selector in the app limits which pages are sent, and therefore how much each run costs.

---

## 6. Tests

```powershell
uv run pytest                     # fast unit + config tests, plus GPU checks (about 10 s)
uv run pytest -m gpu              # only: are torch, Paddle and llama.cpp really using the GPU?
uv run pytest -m integration -s   # every engine on 3 sample pages, checking which device each one used
```

The integration run sends one page each of `ami_organics.pdf`, `sbi_one_page.pdf` and page 1 of `attention_is_all_you_need.pdf`. It makes 3 paid API calls per cloud engine.

---

## 7. What each engine is good at

| Engine       | PDF | Image | Tables | Layout | GPU   | Notes                                         |
|--------------|-----|-------|--------|--------|-------|-----------------------------------------------|
| PaddleOCR    | yes | yes   | okay   | okay   | yes   | Fast, multilingual                            |
| Surya OCR    | yes | yes   | —      | good   | yes   | Modern transformer OCR                        |
| docTR        | yes | yes   | —      | good   | yes   | Torch or TF backend                           |
| Tesseract    | yes | yes   | —      | basic  | no    | CPU, needs system binary                      |
| EasyOCR      | yes | yes   | —      | basic  | yes   | 80+ languages                                 |
| pdfplumber   | yes | no    | good   | text   | no    | Digital PDFs only, no OCR                     |
| Camelot      | yes | no    | best   | tables | no    | Needs Ghostscript                             |
| Docling      | yes | yes   | good   | best   | opt   | IBM, structured output                        |
| Marker       | yes | yes   | good   | best   | opt   | Excellent PDF → Markdown                      |
| OpenAI       | yes | yes   | good   | —      | cloud | Vision LLM, paid per token                    |
| Anthropic    | yes | yes   | good   | —      | cloud | Vision LLM, paid per token                    |
| DeepSeek     | yes | yes   | good   | —      | cloud | Vision LLM, paid per token, cheapest of the 3 |

---

## 8. Troubleshooting

### Cloud engines
- **`OPENAI_API_KEY is not set`** (or the Anthropic or DeepSeek equivalent): add the key to `.env` in the project root.
- **`model_not_found` / 404:** your account may not have access to the default model. Set `*_MODEL` in `.env` to a model you can use.
- **Text cut off on dense pages:** the log (`app.log`) warns `output hit the token limit`. Raise `LLM_OCR_MAX_TOKENS`.
- **Rate-limit (429) errors on long PDFs:** lower `LLM_OCR_CONCURRENCY`, for example to `1`.

### Engines run on CPU although the laptop has an NVIDIA GPU
Run `uv run pytest -m gpu`. Each failing test names the cause:

| Failing test | Cause | Fix |
|---|---|---|
| `test_torch_is_a_cuda_build…` | CPU-only torch was installed (`2.x+cpu`) | `uv sync` |
| `test_a_cuda_llama_server…` | Surya and Marker found only the CPU llama.cpp build | `python scripts\install_llama_cpp.py` |
| `test_paddle_gpu_is_not_forced_off` | `PADDLE_FORCE_CPU=1` is set in your environment | Remove it from Windows *Environment Variables* and restart the terminal |
| `test_paddle_runs_a_kernel_on_the_gpu` | The CPU `paddlepaddle` package is installed | `uv sync` |

Tesseract, pdfplumber and Camelot have no GPU support, so they always run on the CPU.

### Local engines
- **PaddlePaddle install fails** on Python 3.13 or 3.14: use Python 3.11 (`uv venv --python 3.11`).
- **PaddleOCR `NotImplementedError … onednn_instruction.cc:118`:** this is a Windows bug in PaddlePaddle 3.3. The engine works around it with `enable_mkldnn=False`.
- **`OSError: [WinError 127] … torch\lib\shm.dll`:** this happens when `paddle` is imported before `torch`. `engines/logging_setup.py:preload_windows_dlls()` fixes it automatically by loading torch first.
- **Surya returns 0 characters or is very slow:** install the CUDA build of llama.cpp (`python scripts\install_llama_cpp.py --cuda`) and set `LLAMA_CPP_BINARY` to `.tools\llama-cpp\cuda\llama-server.exe`.
- **Surya or Marker is slow the first time:** they download models from HuggingFace on the first run and cache them.
- **Camelot error:** install Ghostscript and restart your terminal.
- **Tesseract not found:** install the binary and add it to `PATH`.
