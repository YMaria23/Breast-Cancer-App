import datetime as dt
import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import config
from .gmic_engine import GmicEngine, breast_bbox, gmic_available


@dataclass
class AnalysisResult:
    malignant: float
    benign: float
    priority: str
    model: str
    latency_s: float
    heatmap: Optional[np.ndarray] = None
    rois: List[dict] = field(default_factory=list)
    explanation: str = ""
    explanation_source: str = ""


def triage_priority(malignant, benign):
    if malignant >= config.HIGH_THRESHOLD:
        return "HIGH"
    if malignant >= config.MEDIUM_MALIGNANT_THRESHOLD or benign >= config.MEDIUM_BENIGN_THRESHOLD:
        return "MEDIUM"
    return "LOW"


class MockBackend:
    name = "Mock v1.2 (random, Milestone 2)"

    def predict(self, image, laterality):
        seed = int(hashlib.md5(np.ascontiguousarray(image).tobytes()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        malignant = rng.uniform(0.025, 0.985)
        benign = rng.uniform(0.025, 0.985)
        h, w = image.shape
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        yy, xx = np.mgrid[0:h, 0:w]
        heatmap = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.06 * w) ** 2))).astype(np.float32)
        size = 0.08 * w
        rois = [{"x": cx - size / 2, "y": cy - size / 2, "w": size, "h": size, "attention": 1.0}]
        return {"malignant": malignant, "benign": benign, "heatmap_malignant": heatmap, "rois": rois}


class GmicBackend:
    def __init__(self, model_indices=(1, 2, 3, 4, 5)):
        self.engine = GmicEngine(model_indices=model_indices)
        self.name = self.engine.version

    def predict(self, image, laterality):
        return self.engine.predict(image, laterality)


def load_env_file(path=os.path.join(config.SOLUTION_DIR, ".env")):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

_backend = None
_backend_lock = threading.Lock()


def backend_name_requested():
    requested = os.environ.get("AI_BACKEND", "").lower()
    if requested in ("mock", "gmic"):
        return requested
    return "gmic" if gmic_available() else "mock"


def get_backend():
    global _backend
    with _backend_lock:
        if _backend is None:
            if backend_name_requested() == "gmic":
                _backend = GmicBackend()
            else:
                _backend = MockBackend()
        return _backend


def set_backend(backend):
    global _backend
    with _backend_lock:
        _backend = backend


def _log(entry):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    with open(os.path.join(config.LOG_DIR, "ai_calls.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def analyse(image: np.ndarray, laterality: str, view: str = "", case_id: str = "", backend=None) -> AnalysisResult:
    backend = backend or get_backend()
    start = time.perf_counter()
    raw = backend.predict(image, laterality)
    latency = time.perf_counter() - start
    result = AnalysisResult(
        malignant=float(raw["malignant"]),
        benign=float(raw["benign"]),
        priority=triage_priority(raw["malignant"], raw["benign"]),
        model=backend.name,
        latency_s=latency,
        heatmap=raw.get("heatmap_malignant"),
        rois=raw.get("rois", []),
    )
    _log({
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "case": case_id,
        "input_sha1": hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest(),
        "input_shape": list(image.shape),
        "laterality": laterality,
        "view": view,
        "model": result.model,
        "malignant": round(result.malignant, 4),
        "benign": round(result.benign, 4),
        "priority": result.priority,
        "latency_s": round(latency, 2),
    })
    return result


def roi_location(roi, bbox, laterality, view):
    x0, y0, x1, y1 = bbox
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    cx = roi["x"] + roi["w"] / 2 - x0
    cy = roi["y"] + roi["h"] / 2 - y0
    distance_from_chest = cx / w if laterality == "L" else 1 - cx / w
    if distance_from_chest < 0.33:
        depth = "posterior, near chest wall"
    elif distance_from_chest < 0.66:
        depth = "middle depth"
    else:
        depth = "anterior, near nipple"
    if view == "MLO":
        vertical = "upper" if cy / h < 0.45 else ("lower" if cy / h > 0.6 else "central")
        return f"{vertical}, {depth}"
    return depth


EXPLAIN_SYSTEM_PROMPT = (
    "You write a short summary of an AI triage result for a radiologist reading a screening mammogram. "
    "Use only the facts given in the input. Do not invent findings, sizes, BI-RADS categories or a diagnosis. "
    "Say what the score means for the reading order, where the model looked, and one limitation. "
    "Write 3 or 4 plain sentences in English, no lists, no headings."
)


def build_facts(result, study):
    view = study.get("view", "")
    laterality = study.get("laterality", "L")
    bbox = study.get("_bbox") or (0, 0, 1, 1)
    top = [roi_location(r, bbox, laterality, view) for r in result.rois[:2]]
    density = {"F": "fatty", "G": "fatty-glandular", "D": "dense-glandular"}.get(study.get("density", ""), "unknown")
    return {
        "view": f"{'right' if laterality == 'R' else 'left'} breast {view}".strip(),
        "breast_density": density,
        "malignancy_score": round(result.malignant, 3),
        "benign_finding_score": round(result.benign, 3),
        "triage_priority": result.priority,
        "triage_rule": "HIGH if malignancy >= 0.10; MEDIUM if malignancy >= 0.03 or benign >= 0.40; else LOW",
        "most_attended_regions": top if study.get("_bbox") else [],
        "model": result.model,
        "model_note": "trained on NYU screening exams; scores are not calibrated for this archive",
    }


def template_explanation(facts):
    regions = facts["most_attended_regions"]
    where = f"The region the model weighted most is {regions[0]}" if regions else "No focal region was localised"
    if facts["triage_priority"] == "HIGH":
        lead = "This image is placed at the top of the worklist"
    elif facts["triage_priority"] == "MEDIUM":
        lead = "This image is flagged for a standard-priority read"
    else:
        lead = "This image is placed at the bottom of the worklist"
    text = (
        f"{lead}: malignancy score {facts['malignancy_score']:.3f}, benign-finding score "
        f"{facts['benign_finding_score']:.3f} ({facts['view']}"
        f"{'' if facts['breast_density'] == 'unknown' else ', ' + facts['breast_density'] + ' tissue'}). "
        f"{where}; see the heatmap and the boxes. "
    )
    if facts["breast_density"] == "dense-glandular":
        text += "Dense tissue can hide lesions, so a low score here is less reassuring. "
    text += "The scores are relative and not calibrated for this archive; the radiologist makes the decision."
    return text


def _ask_llm(client, model, messages, timeout):
    reasoning = os.environ.get("LLM_REASONING", "low")
    arguments = dict(model=model, messages=messages, temperature=0.2, max_tokens=1500, timeout=timeout)
    if reasoning:
        try:
            return client.chat.completions.create(reasoning_effort=reasoning, **arguments)
        except Exception as error:
            if "reasoning" not in str(error).lower() and type(error).__name__ != "BadRequestError":
                raise
    return client.chat.completions.create(**arguments)


def explain(result: AnalysisResult, study: dict, timeout=None) -> tuple:
    facts = build_facts(result, study)
    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    timeout = timeout or float(os.environ.get("LLM_TIMEOUT_S", "60"))
    if not (base_url and api_key and model):
        return template_explanation(facts), "template (no LLM configured)"
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        messages = [
            {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(facts)},
        ]
        response = _ask_llm(client, model, messages, timeout)
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("empty answer")
        _log({"time": dt.datetime.now().isoformat(timespec="seconds"), "case": study.get("accession", ""),
              "llm_model": model, "llm_input": facts, "llm_output": text})
        return text, f"LLM: {model}"
    except Exception as error:
        _log({"time": dt.datetime.now().isoformat(timespec="seconds"), "case": study.get("accession", ""),
              "llm_model": model, "llm_error": f"{type(error).__name__}: {error}"[:500]})
        return template_explanation(facts), f"template (LLM failed: {type(error).__name__})"


def guess_laterality(image):
    x0, _, x1, _ = breast_bbox(image.astype(np.float32))
    pixels = image.astype(np.float32)
    middle = (x0 + x1) // 2
    left = pixels[:, x0:middle].sum()
    right = pixels[:, middle:x1].sum()
    return "L" if left >= right else "R"
