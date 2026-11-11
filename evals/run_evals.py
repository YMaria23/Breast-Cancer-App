import argparse
import csv
import datetime as dt
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "software-solution"))

from ai import ai_service


def view_info(code):
    code = (code or "").strip().upper()
    view = "MLO" if "MLO" in code else "CC"
    side = "R" if code.endswith("RT") and "LRT" not in code else "L"
    return view, side

ACCEPTED = {"M": {"HIGH", "MEDIUM"}, "B": {"MEDIUM", "LOW"}, "NORM": {"LOW"}}


def auc(positives, negatives):
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def load_image(images_dir, image_id):
    for extension in (".png", ".jpg"):
        path = os.path.join(images_dir, image_id + extension)
        if os.path.isfile(path):
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    raise FileNotFoundError(f"{image_id} not found in {images_dir}")


def evaluate(backend, cases, images_dir):
    rows = []
    for case in cases:
        view, side = view_info(case["view"])
        image = load_image(images_dir, case["image"])
        result = ai_service.analyse(image, side, view, f"eval:{case['case']}", backend=backend)
        rows.append({
            **case,
            "model": backend.name,
            "malignant": round(result.malignant, 4),
            "benign": round(result.benign, 4),
            "priority": result.priority,
            "pass": result.priority in ACCEPTED[case["truth"]],
            "latency_s": round(result.latency_s, 2),
        })
        print(f"{backend.name:32s} {case['case']} {case['image']} truth={case['truth']:4s} "
              f"priority={result.priority:6s} malignant={result.malignant:.3f}", flush=True)
    return rows


def summary(rows):
    malignant = [r["malignant"] for r in rows if r["truth"] == "M"]
    rest = [r["malignant"] for r in rows if r["truth"] != "M"]
    abnormal = [r["malignant"] for r in rows if r["truth"] != "NORM"]
    normal = [r["malignant"] for r in rows if r["truth"] == "NORM"]
    flagged_m = sum(r["priority"] in ("HIGH", "MEDIUM") for r in rows if r["truth"] == "M")
    low_normal = sum(r["priority"] == "LOW" for r in rows if r["truth"] == "NORM")
    return {
        "score": sum(r["pass"] for r in rows),
        "total": len(rows),
        "sensitivity": f"{flagged_m} / {len(malignant)}",
        "normals_low": f"{low_normal} / {len(normal)}",
        "auc_malignant": auc(malignant, rest),
        "auc_abnormal": auc(abnormal, normal),
        "latency": sum(r["latency_s"] for r in rows) / max(1, len(rows)),
    }


def write_markdown(path, cases, results, images_dir):
    names = list(results)
    lines = [
        "# Eval results",
        "",
        f"Run on {dt.datetime.now():%Y-%m-%d %H:%M} with `python evals/run_evals.py`. Images: `{images_dir}`.",
        "",
        "A case passes when the AI priority is in the accepted set for its ground truth: "
        "malignant → HIGH or MEDIUM, benign → MEDIUM or LOW, normal → LOW.",
        "",
        "| Case | Image | View | Density | Truth | Expected | " + " | ".join(names) + " |",
        "|---|---|---|---|---|---|" + "---|" * len(names),
    ]
    for i, case in enumerate(cases):
        cells = []
        for name in names:
            r = results[name][i]
            mark = "pass" if r["pass"] else "FAIL"
            cells.append(f"{r['priority']} ({r['malignant']:.3f}) {mark}")
        lines.append(f"| {case['case']} | {case['image']} | {case['view']} | {case['density']} | {case['truth']} | "
                     f"{case['expected_priority']} | " + " | ".join(cells) + " |")
    lines += ["", "## Summary", "", "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    summaries = {name: summary(rows) for name, rows in results.items()}
    for label, key, fmt in [
        ("Cases passed", None, None),
        ("Malignant flagged (HIGH or MEDIUM)", "sensitivity", "{}"),
        ("Normals placed LOW", "normals_low", "{}"),
        ("AUC, malignant vs rest", "auc_malignant", "{:.2f}"),
        ("AUC, any lesion vs normal", "auc_abnormal", "{:.2f}"),
        ("Mean time per image (s)", "latency", "{:.1f}"),
    ]:
        if key is None:
            values = [f"{s['score']} / {s['total']}" for s in summaries.values()]
        else:
            values = [fmt.format(s[key]) for s in summaries.values()]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run the mock and the real model on the eval cases")
    parser.add_argument("--images-dir", default=os.path.join(os.path.dirname(HERE), "software-solution", "archive", "DMID_PNG", "1024", "TIFF"))
    parser.add_argument("--cases", default=os.path.join(HERE, "cases.csv"))
    parser.add_argument("--backends", default="mock,gmic")
    args = parser.parse_args()
    with open(args.cases, newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    results = {}
    for name in args.backends.split(","):
        backend = ai_service.MockBackend() if name == "mock" else ai_service.GmicBackend()
        results[backend.name] = evaluate(backend, cases, args.images_dir)
    with open(os.path.join(HERE, "results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(results.values()))[0].keys()))
        writer.writeheader()
        for rows in results.values():
            writer.writerows(rows)
    write_markdown(os.path.join(HERE, "RESULTS.md"), cases, results, args.images_dir)
    for name, rows in results.items():
        s = summary(rows)
        print(f"{name}: {s['score']}/{s['total']} passed, AUC malignant {s['auc_malignant']:.2f}, "
              f"AUC abnormal {s['auc_abnormal']:.2f}")


if __name__ == "__main__":
    main()
