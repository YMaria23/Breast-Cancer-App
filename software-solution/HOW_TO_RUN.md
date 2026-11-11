# How to run the app (Milestone 3)

The app is `ui_ppi.py`: a PACS-style viewer that opens a mammogram, runs the GMIC model on it, shows the urgency, the scores and a heatmap, and writes a short explanation with an LLM on UBB LLMHub. Details about the models are in `../MODEL.md`.

## Requirements

- **Python 3.12** (3.10–3.12 work; 3.14 is too new for some packages)
- About 2 GB of disk for PyTorch, plus 300 MB for the GMIC weights
- Git
- Optional: a UBB Microsoft account for LLMHub (for the LLM explanation)

Python packages are in `requirements.txt`: torch, torchvision, numpy, opencv-python, PyQt5, pydicom, pandas, openpyxl, openai.

## 1. Set up (once)

### Windows (cmd)

```
cd /d C:\path\to\Breast-Cancer-App\software-solution
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
git clone https://github.com/nyukat/GMIC third_party/GMIC
```

### Linux / macOS

```
cd path/to/Breast-Cancer-App/software-solution
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
git clone https://github.com/nyukat/GMIC third_party/GMIC
```

On Linux, install the CPU build of PyTorch first (the line above); otherwise pip downloads the much larger GPU build.

GMIC is AGPL-3.0, so it is not part of our repo. `third_party/` is in `.gitignore`.

## 2. Connect LLMHub (once, optional)

1. Sign in to LLMHub with your UBB Microsoft account and open **Connect a client**.
2. Copy `.env.example` to `.env` (same folder) and fill in your own values:

```
LLM_BASE_URL=the API base URL from "Connect a client"
LLM_API_KEY=your own key
LLM_MODEL=qwen3.8-flash-next
```

3. Test it:

```
python -m ai.check_llm
```

It should end with `LLM connection works.`

`.env` is git-ignored. Never commit your key and never share it; each team member uses their own.

Without `.env` the app still works; the explanation then comes from a fixed template.

## 3. Run

```
cd /d C:\path\to\Breast-Cancer-App\software-solution
.venv\Scripts\activate
python ui_ppi.py
```

(On Linux: `source .venv/bin/activate` instead of the second line.)

1. **Upload Image (.dcm / .png)**: for example `archive\DMID_PNG\1024\TIFF\IMG491.png` (a malignant case the model marks HIGH).
2. **Run AI Analysis**: the first run takes about 15 s (it loads the model), later runs about 7 s, plus 3–4 s for the LLM explanation.
3. **Show Heatmap**: turns the model's attention map and its 3 strongest regions on or off (orange box = strongest).
4. **Good / Bad**: saved to `logs/feedback.jsonl`.

Every model and LLM call is logged in `logs/ai_calls.jsonl`.

## 4. Run the tests

From the repository root (one level up), with the virtual environment active:

```
python evals/run_evals.py
```

It runs the mock and GMIC on the 22 cases in `evals/cases.csv` and rewrites `evals/RESULTS.md` (about 3 minutes).

## Settings

All optional, in `.env` or as environment variables:

| Variable | Default | What it does |
|---|---|---|
| `AI_BACKEND` | `gmic` if the weights exist, else `mock` | Force `mock` or `gmic` |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | none | LLM for the explanation |
| `LLM_TIMEOUT_S` | 60 | Seconds before falling back to the template |
| `LLM_REASONING` | low | Thinking level for models that support it (empty to turn off) |
| `AI_TIMEOUT_S` | 120 | Seconds before the UI shows the "no answer" message |
| `GMIC_DIR` | `third_party/GMIC` | Where the GMIC repo is |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Failed to launch 'C:\Python314\python.exe'` | The default Python is broken. Use `py -3.12 -m venv .venv` as above; inside the venv, `python` is 3.12. |
| Result says `Source: Mock v1.2` | GMIC was not found. Check that `third_party/GMIC/models/sample_model_1.p` exists. |
| `Explanation by: template (LLM failed: APITimeoutError)` | The LLM did not answer in time. Run `python -m ai.check_llm`; if only one model times out, change `LLM_MODEL` in `.env`. |
| `Explanation by: template (no LLM configured)` | `.env` is missing or not in `software-solution/`. On Windows check it is not called `.env.txt`. |
| `ModuleNotFoundError` | The venv is not active. Run `.venv\Scripts\activate` first. |
