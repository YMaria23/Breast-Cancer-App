import os
import sys
import threading

import cv2
import numpy as np

GMIC_HEIGHT = 2944
GMIC_WIDTH = 1920
PERCENT_T = {1: 0.02, 2: 0.03, 3: 0.03, 4: 0.05, 5: 0.1}
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_gmic_dir():
    return os.environ.get("GMIC_DIR") or os.path.join(PROJECT_ROOT, "third_party", "GMIC")


def gmic_available(gmic_dir=None):
    gmic_dir = gmic_dir or default_gmic_dir()
    return os.path.isfile(os.path.join(gmic_dir, "models", "sample_model_1.p"))


def breast_bbox(image):
    h, w = image.shape
    scale = 512.0 / max(h, w)
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    small = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return 0, 0, w, h
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, _ = stats[largest]
    pad = 4
    x0 = max(0, int((x - pad) / scale))
    y0 = max(0, int((y - pad) / scale))
    x1 = min(w, int((x + bw + pad) / scale))
    y1 = min(h, int((y + bh + pad) / scale))
    return x0, y0, x1, y1


class GmicEngine:
    def __init__(self, gmic_dir=None, model_indices=(1, 2, 3, 4, 5), threads=None):
        import torch

        self.torch = torch
        self.gmic_dir = gmic_dir or default_gmic_dir()
        if not gmic_available(self.gmic_dir):
            raise FileNotFoundError(
                f"GMIC weights not found in {self.gmic_dir}. "
                "Run: git clone https://github.com/nyukat/GMIC third_party/GMIC"
            )
        if threads:
            torch.set_num_threads(threads)
        if self.gmic_dir not in sys.path:
            sys.path.insert(0, self.gmic_dir)
        from src.modeling import gmic as gmic_module

        self.models = []
        for index in model_indices:
            parameters = {
                "device_type": "cpu",
                "gpu_number": 0,
                "max_crop_noise": (100, 100),
                "max_crop_size_noise": 100,
                "cam_size": (46, 30),
                "K": 6,
                "crop_shape": (256, 256),
                "post_processing_dim": 256,
                "num_classes": 2,
                "use_v1_global": False,
                "percent_t": PERCENT_T[index],
            }
            model = gmic_module.GMIC(parameters)
            path = os.path.join(self.gmic_dir, "models", f"sample_model_{index}.p")
            state = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(state, strict=False)
            model.eval()
            self.models.append(model)
        self.lock = threading.Lock()
        self.version = f"GMIC (NYU) ensemble of {len(self.models)}"

    def preprocess(self, image, laterality):
        image = image.astype(np.float32)
        flipped = laterality == "R"
        if flipped:
            image = np.fliplr(image)
        x0, y0, x1, y1 = breast_bbox(image)
        crop = image[y0:y1, x0:x1]
        scale = min(GMIC_HEIGHT / crop.shape[0], GMIC_WIDTH / crop.shape[1])
        new_w = max(1, int(round(crop.shape[1] * scale)))
        new_h = max(1, int(round(crop.shape[0] * scale)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((GMIC_HEIGHT, GMIC_WIDTH), np.float32)
        offset_y = (GMIC_HEIGHT - new_h) // 2
        canvas[offset_y:offset_y + new_h, :new_w] = resized
        canvas -= canvas.mean()
        canvas /= max(canvas.std(), 1e-5)
        geometry = {
            "shape": image.shape,
            "flipped": flipped,
            "box": (x0, y0, x1, y1),
            "scale": scale,
            "offset_y": offset_y,
            "size": (new_w, new_h),
        }
        return canvas, geometry

    def _map_saliency(self, saliency, geometry):
        h, w = geometry["shape"]
        x0, y0, x1, y1 = geometry["box"]
        new_w, new_h = geometry["size"]
        oy = geometry["offset_y"]
        full = cv2.resize(saliency.astype(np.float32), (GMIC_WIDTH, GMIC_HEIGHT), interpolation=cv2.INTER_CUBIC)
        region = full[oy:oy + new_h, :new_w]
        region = cv2.resize(region, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        out = np.zeros((h, w), np.float32)
        out[y0:y1, x0:x1] = np.clip(region, 0, 1)
        if geometry["flipped"]:
            out = np.fliplr(out)
        return np.ascontiguousarray(out)

    def _map_rois(self, locations, attentions, geometry):
        h, w = geometry["shape"]
        x0, y0, _, _ = geometry["box"]
        scale = geometry["scale"]
        oy = geometry["offset_y"]
        size = 256 / scale
        rois = []
        for (top, left), attention in zip(locations, attentions):
            x = left / scale + x0
            y = (top - oy) / scale + y0
            if geometry["flipped"]:
                x = w - x - size
            rois.append({
                "x": float(max(0, x)),
                "y": float(max(0, y)),
                "w": float(size),
                "h": float(size),
                "attention": float(attention),
            })
        rois.sort(key=lambda r: -r["attention"])
        return rois

    def predict(self, image, laterality):
        canvas, geometry = self.preprocess(image, laterality)
        batch = self.torch.from_numpy(canvas[None, None].copy())
        benign, malignant = [], []
        saliency = None
        rois = []
        with self.lock, self.torch.no_grad():
            for i, model in enumerate(self.models):
                output = model(batch).cpu().numpy()
                benign.append(float(output[0, 0]))
                malignant.append(float(output[0, 1]))
                if i == 0:
                    saliency = model.saliency_map.cpu().numpy()[0]
                    rois = self._map_rois(
                        model.patch_locations[0],
                        model.patch_attns[0].cpu().numpy(),
                        geometry,
                    )
        return {
            "benign": float(np.mean(benign)),
            "malignant": float(np.mean(malignant)),
            "heatmap_malignant": self._map_saliency(saliency[1], geometry),
            "heatmap_benign": self._map_saliency(saliency[0], geometry),
            "rois": rois,
        }
