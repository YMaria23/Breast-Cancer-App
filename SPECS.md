# SPECS: AI-Assisted Breast Cancer Detection

> Challenge 2, AI for women's health: breast cancer detection
> Challenge owner: Dr. Andrei Roman

## Problem statement

**Hospitals** have trouble **identifying early-stage breast cancer** in the step **after a mammography and before a biopsy**.
Our solution will **facilitate the detection of breast cancer**, and help the local medical industry grow and raise awareness about screening.

## Context

- Radiologists analyse a high volume of mammograms every day, which raises the risk of fatigue-related errors.
- Advanced AI models exist, but there are still cases where their accuracy is not good enough, especially in dense breast tissue.
- Commercial AI tools are expensive and price out smaller clinics.
- Getting a large dataset of annotated mammograms is hard because of patient privacy rules.

## Target users

| Who | Role |
|---|---|
| Hospitals, imaging centres | Customers (they pay) |
| Radiologists, doctors | Daily users |
| Researchers | Use it on retrospective data and public datasets |

**When they need it most:** right before a biopsy, when a doctor needs a second opinion without ordering more tests, in clinics without a breast specialist, and on busy screening days with a backlog.

## What we learned from past solutions

**Worked well:**
- Early CAD systems flagged high-contrast microcalcifications and obvious clustered lesions.
- Early systems sorted normal mammograms quickly, so high-risk cases reached the doctor faster.
- Open datasets (DDSM, CBIS-DDSM, INbreast) let academic models reach strong benchmark results.

**Did not work well:**
- Models flagged too many benign spots, so radiologists started ignoring the warnings (alert fatigue).
- Doctors found the software hard to learn.
- Models trained on one scanner vendor performed much worse on other vendors' machines.
- Dense breast tissue hid tumours or caused false alarms that simple algorithms could not tell apart.

## Goals

- Reduce false positives and alert fatigue without lowering detection performance
- Improve detection in dense breast tissue
- Automatically triage normal screening exams
- Minimise false negatives for small lesions
- Identify subtle microcalcifications
- Decrease the time a radiologist spends per case

## MVP features

The MVP is a **research prototype on public data, labelled "not for clinical use"**.

1. **Upload** a mammography exam.
2. **Detection:** suspicion score for each image.
3. **Heatmap** over the mammogram showing where the model is looking.
4. **Triage worklist:** exams sorted by suspicion, likely-normal exams at the bottom, urgent ones on top.
5. **Explanation:** a short plain-language summary of the findings, written by an LLM grounded in screening guidelines.
6. **Simple interface** that needs no training to use.

**Out of scope for now:** real patient data, PACS integration, ultrasound/MRI, training our own model from scratch.

## Tech stack

| Part | Choice |
|---|---|
| Detection + heatmaps | **GMIC** (NYU), pretrained, runs on CPU |
| Optional 5-year risk | **Mirai** (MIT) |
| Explanations | **Gemma 4 26B** or **Qwen3.5-35B-A3B**, 4-bit, served with llama.cpp / Ollama |
| RAG | **Qwen3-Embedding-0.6B** over screening guidelines and the owner's teaching cases |
| Data | **CBIS-DDSM**, **INbreast** |
| Hosting | University CPU server |

The detection model sits behind one interface with three implementations: a **mock** (to build the UI now), **GMIC** (next), and the **MIRPR model** (when it is ready). Swapping models does not change the rest of the app.

**For Demo Day:** precompute heatmaps for the demo cases, stream the LLM text, and warm the models up before our slot.

## Competitors

| Competitor | What they do |
|---|---|
| Lunit, ScreenPoint Transpara, Kheiron Mia, Vara | Certified AI for mammography |

What they are doing: getting FDA/CE clearances, integrating directly into hospital PACS so radiologists don't switch screens, and adding real-time alert triage.

**Our opportunities:**
- AI focused on dense breast tissue
- Affordable for smaller clinics that the big vendors price out
- Explainable results (heatmap + written reason) to build trust
- Runs on local hardware, so images stay in the hospital and in the EU
- Later: direct PACS integration and triage

## Trends

| Against us | In our favour |
|---|---|
| Strict data privacy (GDPR) | FDA/CE approvals now favour AI triage |
| Radiologist burnout from old "noisy" CAD makes doctors sceptical | More diverse multi-centre datasets available |
| EU AI Act (high-risk AI rules) | Growing radiologist shortage creates demand |

## Regulation

- **Now:** research prototype on public data, not a medical device.
- **Later:** a clinical second reader is a medical device under **EU MDR Rule 11** (class IIa or IIb) and **high-risk AI under the EU AI Act** (from August 2028). The path is a retrospective study at the owner's hospital, then a prospective one.
