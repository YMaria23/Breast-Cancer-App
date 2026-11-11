# MODEL.md

Milestone 3: the mocked AI from Milestone 2 is replaced by a real model. The UI (`software-solution/ui_ppi.py`, same layout as Milestone 2) now calls one function with a fixed input and output:

```python
from ai import ai_service
result = ai_service.analyse(image, laterality, view)
```

`image` is a 2D grayscale mammogram (numpy array), `laterality` is `"L"` or `"R"`, `view` is `"CC"` or `"MLO"`. The result has `malignant`, `benign`, `priority` (HIGH / MEDIUM / LOW), `heatmap`, `rois`, `model` and `latency_s`. The backend behind it is chosen with the `AI_BACKEND` environment variable (`gmic` or `mock`); the UI does not change.

## 1. Detection model: GMIC

**What it is.** GMIC (Globally-Aware Multiple Instance Classifier) from the NYU Langone breast imaging group (Shen et al., *Medical Image Analysis*, 2021). It looks at the whole mammogram at low resolution, picks the 6 most suspicious patches, looks at them at full resolution, and fuses both views of the image into two probabilities: benign finding and malignant finding. It also outputs a saliency map, which we show as the heatmap, and the 6 patch locations, which we show as the numbered boxes.

**Why this model.**

- It matches the MVP in SPECS.md: a suspicion score per image plus a heatmap showing where the model looked.
- It is pretrained on a very large screening set (NYU Breast Cancer Screening Dataset, about 1 million images), so we can use it as-is. This is step 1 of the "four ways" ladder from Lab 3: use a ready-made model before fine-tuning or training.
- The weights are public (5 checkpoints in the GitHub repo) and it runs on a laptop CPU.
- It already was the "next" model in our SPECS.md tech stack.

**How we use it.** All 5 published checkpoints, averaged (the ensemble the authors recommend). Heatmap and boxes come from checkpoint 1.

**Preprocessing we added** (GMIC expects NYU-style 2944 x 1920 crops):

1. Flip right-breast images so the chest wall is on the left, like GMIC's training data.
2. Find the breast with an Otsu threshold and keep the largest connected region.
3. Crop to the breast, resize it to fit 2944 x 1920 without stretching, place it against the left edge.
4. Standardise (mean 0, std 1), the same as GMIC's own loader.
5. Map the heatmap and boxes back to the original image so they line up in the viewer.

**Triage rule** (in `software-solution/ai/config.py`):

| Priority | Rule |
|---|---|
| HIGH | malignant ≥ 0.10 |
| MEDIUM | malignant ≥ 0.03, or benign ≥ 0.40 |
| LOW | everything else |

These cut-offs were chosen by looking at the 22 eval cases, so the eval score below is optimistic. Before Demo Day we should pick them on a separate set of DMID images and keep the eval set only for testing.

## 2. Explanation text: Qwen3.8 Flash Next on UBB LLMHub

The "Why" text in the result box is written by an LLM, which is step 1 of the ladder (a prompt, no training).

**Which model and why.** `qwen3.8-flash-next` on **LLMHub**, the faculty's GPU platform, used **text-only**.
- LLMHub needs no VPN (sign-in with the UBB Microsoft account), runs on GPU, and keeps data inside the faculty.
- We tested both LLMHub text models on 3 November 2026 with `python -m ai.check_llm`. Qwen answered a short test in 0.7 s and wrote a full explanation in 3.6 s. `gpt-oss-20b` listed as available but did not answer within 60 s, even with low reasoning. We keep gpt-oss as an alternative and will compare both on the same cases when it is back.
- We do **not** use Qwen's image input on the mammogram. A general vision model is not trained to read mammograms, so it would add a second, untested "opinion". GMIC stays the only model that looks at the image; Qwen only turns GMIC's numbers into a sentence.
- Backup: the faculty AI Server (`gpt-oss-20b` on CPU, through the faculty VPN, team key). Switching is only a change in `.env`.

**What it gets.** A small JSON with the two scores, the triage priority, the view (e.g. "left breast MLO"), the density if known, and up to two region descriptions (e.g. "upper, posterior, near chest wall"). The system prompt tells it to use only these facts, not to invent findings, sizes, BI-RADS categories or a diagnosis, and to write 3–4 plain sentences.

**Setup.** Copy `software-solution/.env.example` to `software-solution/.env` (git-ignored) and fill in the values from LLMHub's "Connect a client" page:

```
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=qwen3.8-flash-next
```

Check the connection with `python -m ai.check_llm` (from `software-solution/`).

**When it fails.** If `.env` is missing, or the call fails or takes more than 60 s (`LLM_TIMEOUT_S`), the app writes the same kind of text from a fixed template, so the result box is never empty. The result box shows who wrote the text ("Explanation by: LLM: qwen3.8-flash-next" or "template"). Every LLM input, output and error is logged in `software-solution/logs/ai_calls.jsonl`.

## 3. Licences

| Part | Licence | What it means for us |
|---|---|---|
| GMIC code and weights | **GNU AGPL-3.0** | Fine for a public university repo. If we ever distribute the app or run it as a network service, our whole app must be released under AGPL too. A commercial product would need the MIRPR model or another permissively licensed one. We do **not** copy GMIC into our repo; it is cloned into `software-solution/third_party/` (git-ignored). |
| DMID dataset | Public research dataset (figshare) | Used only as demo data. Check the exact licence and citation on the dataset page before Demo Day and cite it on the slides. |
| PyQt5 | GPL-3.0 (or commercial) | Same note as AGPL: fine for the course, a commercial version would need a Qt licence or PySide6 (LGPL). |
| PyTorch, OpenCV, NumPy, pydicom | BSD / Apache / MIT | No restrictions for us. |
| `openai` Python library | Apache-2.0 | Only a client library; works with LLMHub, the AI Server, Groq, Ollama, etc. |
| Qwen3.8 Flash Next (on LLMHub) | Open weights, hosted by the faculty | Check the exact licence on the model card before Demo Day. We only send it text. |
| gpt-oss-20b (LLMHub / AI Server, alternative) | Apache-2.0 | No restrictions for us. |

## 4. Where it runs

- **Locally**, on the radiologist's computer, CPU only. No GPU needed.
- Loading the 5 checkpoints: about 7 s, done once on the first analysis.
- About 6–7 s per image on 2 CPU cores; faster on a normal laptop with more cores. Peak memory about 1.3 GB.
- The analysis runs in a background thread, so the UI never freezes. If there is no answer after 120 s (`AI_TIMEOUT_S`), the UI shows the error state and the radiologist can keep reading.

## 5. What data leaves the app

- **Images never leave the computer.** GMIC runs locally.
- To LLMHub (only when `.env` is set) goes only the small JSON described in section 2: scores, priority, view, density, region descriptions. No name, no patient ID, no file name, no pixels. This follows the LLMHub rule of no real patient data.
- The test images come from the public DMID dataset; DMID PNGs carry no patient data. The UI shows patient name and ID only from DICOM files the user opens, and never sends them anywhere. **No real patient data** is used anywhere, and none may be sent to free APIs.

## 6. Results on the test set

See `evals/RESULTS.md` (22 cases: 7 malignant, 7 benign, 8 normal, from DMID; ground truth is DMID's radiologist annotation).

| | Mock (Milestone 2) | GMIC ensemble |
|---|---|---|
| Cases passed | 8 / 22 | **16 / 22** |
| Malignant flagged | 7 / 7 | 3 / 7 |
| Normals placed LOW | 0 / 8 | 7 / 8 |
| AUC malignant vs rest | 0.49 | 0.62 |
| AUC any lesion vs normal | 0.58 | 0.75 |

The mock "flags" every malignant case only because it flags almost everything at random; it puts no normal case at the bottom. GMIC is clearly better at sorting normals to the bottom of the worklist, but it **misses 4 of 7 malignant cases**. That is the number to improve.

## 7. Known limitations

- **Domain shift.** GMIC was trained on 16-bit NYU images from one vendor. Our DMID copies are 8-bit PNGs resized to 1024 x 1024 (the aspect ratio is squashed). This is the "trained on one scanner, worse on another" problem from SPECS.md, and it is the most likely reason for the missed cancers.
- **Scores are not calibrated** for this archive. A malignancy score of 0.2 is high here, which would be low on NYU data. The UI says so.
- **One image at a time.** Real screening reads 4 views per exam and compares left with right. DMID gives single images.
- **Small test set** (22 cases). The AUCs have wide error bars.

## 8. Next steps

1. Run `python evals/run_evals.py` on a larger set (the script takes any folder; all 510 DMID images take about an hour on a laptop) and pick the triage cut-offs on a separate split.
2. Ask Dr. Roman to review the 22 expected answers and add cases he cares about (dense tissue, small calcifications).
3. Swap in the MIRPR model when it is ready: write a new backend class with the same `predict(image, laterality)` method, set `AI_BACKEND`, rerun the evals, and keep it only if it beats GMIC's score.
4. Optional: try the team's Mammo-CLIP fine-tune (AUC 0.67 on CBIS-DDSM) as another backend.

## How to run

Full instructions, including setup on Windows and Linux and troubleshooting: `software-solution/HOW_TO_RUN.md`. In short:

From `software-solution/`:

```
pip install -r requirements.txt
git clone https://github.com/nyukat/GMIC third_party/GMIC
python ui_ppi.py
```

If pip installs the large CUDA build of PyTorch, use the CPU one: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`. Without the GMIC folder the app falls back to the mock. Set `AI_BACKEND=mock` to force the mock.

Upload a `.dcm` or `.png` (e.g. from `archive/DMID_PNG/1024/TIFF`), click "Run AI Analysis". The first run loads the model (about 7 s), later runs take about 6–7 s. "Show Heatmap" toggles the model's attention map and the 3 regions it weighted most (orange box = strongest). Laterality comes from the DICOM header or is detected from the image; "Good" / "Bad" feedback is logged to `logs/feedback.jsonl`.

To rerun the tests (from the repository root): `python evals/run_evals.py`.
