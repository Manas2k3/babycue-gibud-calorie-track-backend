"""
Core multi-model YOLO food detection ensemble logic.
"""

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import (
    MODELS_ROOT,
    DOMAIN_WEIGHT_OVERRIDES,
    PER_MODEL_CONF_THRESH,
    WBF_IOU_THR,
    FINAL_CONF_THRESH,
    NON_FOOD_CLASSES,
    DISABLED_DOMAINS,
    MIN_MODELS_FOR_CONFIRMATION,
)
from nutrition import NutritionLookup
from pipeline.classifier_verification import ClassifierVerifier, normalize_label


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[\s_-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def build_master_class_list(models):
    seen = {}
    for m in models:
        for _, raw_name in m.names.items():
            norm = normalize_name(raw_name)
            if norm in NON_FOOD_CLASSES:
                continue
            if norm not in seen:
                seen[norm] = raw_name
    master_names = sorted(seen.keys())
    name_to_idx = {n: i for i, n in enumerate(master_names)}
    display_names = [seen[n] for n in master_names]
    return name_to_idx, display_names


def discover_models(models_root: Path):
    root = Path(models_root)
    if not root.is_dir():
        raise FileNotFoundError(f"MODELS_ROOT '{models_root}' not found")

    paths, domain_names = [], []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        pt_files = list(sub.glob("*.pt"))
        if not pt_files:
            continue
        paths.append(str(pt_files[0]))
        domain_names.append(sub.name)
    return paths, domain_names


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _draw_boxes(img, boxes_px, scores, names, color):
    vis = img.copy()
    for (x1, y1, x2, y2), score, name in zip(boxes_px, scores, names):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def build_montage(tiles, out_path, tile_w=340, tile_h=280, cols=3, bar_h=28):
    n = len(tiles)
    rows = (n + cols - 1) // cols
    cell_h = tile_h + bar_h
    canvas = np.full((rows * cell_h, cols * tile_w, 3), 30, dtype=np.uint8)

    for i, (label, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        resized = cv2.resize(img, (tile_w, tile_h))
        y0, x0 = r * cell_h, c * tile_w
        cv2.rectangle(canvas, (x0, y0), (x0 + tile_w, y0 + bar_h), (45, 45, 45), -1)
        cv2.putText(canvas, label, (x0 + 8, y0 + bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[y0 + bar_h: y0 + bar_h + tile_h, x0: x0 + tile_w] = resized

    cv2.imwrite(str(out_path), canvas)
    return canvas


def _to_native(obj):
    """Recursively convert numpy scalar/array types to native Python types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class FoodEnsemble:
    def __init__(self, models_root: Optional[Path] = None):
        self.models_root = Path(models_root) if models_root else MODELS_ROOT
        self.model_paths, self.domain_names = discover_models(self.models_root)

        if not self.model_paths:
            raise RuntimeError(
                f"No trained .pt models found under '{self.models_root}'. "
                f"Expected one subfolder per domain, each containing a .pt file."
            )

        self.models = [YOLO(p) for p in self.model_paths]
        self.model_weights = [
            DOMAIN_WEIGHT_OVERRIDES.get(d, 1.0) for d in self.domain_names
        ]
        self.active_mask = [d not in DISABLED_DOMAINS for d in self.domain_names]
        self.name_to_idx, self.display_names = build_master_class_list(self.models)
        self.nutrition = NutritionLookup()
        self.classifier_verifier = ClassifierVerifier()

    def info(self) -> dict:
        return {
            "domains": self.domain_names,
            "weights": dict(zip(self.domain_names, self.model_weights)),
            "unified_classes": self.display_names,
        }

    def _run_raw(self, img: np.ndarray):
        h, w = img.shape[:2]
        boxes_list, scores_list, labels_list = [], [], []
        per_model_raw = []

        for model, domain in zip(self.models, self.domain_names):
            results = model.predict(source=img, conf=PER_MODEL_CONF_THRESH, verbose=False)[0]

            boxes, scores, labels = [], [], []
            raw_boxes_px, raw_scores, raw_names = [], [], []

            if results.boxes is not None and len(results.boxes) > 0:
                xyxy = results.boxes.xyxy.cpu().numpy()
                conf = results.boxes.conf.cpu().numpy()
                cls = results.boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), c, cid in zip(xyxy, conf, cls):
                    raw_name = model.names[int(cid)]
                    norm_name = normalize_name(raw_name)
                    if norm_name not in self.name_to_idx:
                        continue
                    master_id = self.name_to_idx[norm_name]

                    boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
                    scores.append(float(c))
                    labels.append(master_id)

                    raw_boxes_px.append([x1, y1, x2, y2])
                    raw_scores.append(float(c))
                    raw_names.append(raw_name)

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
            per_model_raw.append((domain, raw_boxes_px, raw_scores, raw_names))

        return boxes_list, scores_list, labels_list, per_model_raw

    def _fuse(self, boxes_list, scores_list, labels_list):
        """
        Custom weighted merge - NOT standard Weighted Boxes Fusion's
        agreement-penalized average.

        Your 6 domains are non-overlapping SPECIALISTS (soft drinks vs
        Indian food vs western food), not redundant generalist voters all
        looking at the same universal class space. Standard WBF divides a
        detection's confidence down based on how many of the TOTAL active
        models also detected it - which silently destroys a genuinely
        correct 92% detection from your one relevant domain expert, just
        because two structurally unrelated domains (e.g. soft drinks) had
        nothing to say about a plate of food.

        Instead: cluster overlapping boxes of the SAME class across active
        domains (by IoU), and within each cluster take the domain-weighted
        MAX confidence (no division by total model count) plus a
        confidence-weighted average of the box coordinates.
        """
        flat = []
        for boxes, scores, labels, weight, active in zip(
            boxes_list, scores_list, labels_list, self.model_weights, self.active_mask
        ):
            if not active:
                continue
            for box, score, label in zip(boxes, scores, labels):
                flat.append({"box": box, "score": float(score) * weight, "label": label})

        if not flat:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

        flat.sort(key=lambda d: -d["score"])
        clusters = []
        for det in flat:
            placed = False
            for cluster in clusters:
                if cluster[0]["label"] == det["label"] and _iou(cluster[0]["box"], det["box"]) >= WBF_IOU_THR:
                    cluster.append(det)
                    placed = True
                    break
            if not placed:
                clusters.append([det])

        fused_boxes, fused_scores, fused_labels = [], [], []
        for cluster in clusters:
            best_score = max(d["score"] for d in cluster)
            total_w = sum(d["score"] for d in cluster)
            avg_box = [
                sum(d["box"][i] * d["score"] for d in cluster) / total_w
                for i in range(4)
            ]
            fused_boxes.append(avg_box)
            fused_scores.append(min(best_score, 1.0))  # domain weight >1.0 could push over 1.0
            fused_labels.append(cluster[0]["label"])

        fused_boxes = np.array(fused_boxes) if fused_boxes else np.empty((0, 4))
        fused_scores = np.array(fused_scores) if fused_scores else np.empty((0,))
        fused_labels = np.array(fused_labels) if fused_labels else np.empty((0,))

        keep = fused_scores >= FINAL_CONF_THRESH
        return fused_boxes[keep], fused_scores[keep], fused_labels[keep]

    def predict(self, img: np.ndarray, run_dir: Path, save_visuals: bool = True) -> dict:
        h, w = img.shape[:2]
        boxes_list, scores_list, labels_list, per_model_raw = self._run_raw(img)
        fused_boxes, fused_scores, fused_labels = self._fuse(boxes_list, scores_list, labels_list)

        result = {
            "image_size": {"width": w, "height": h},
            "raw_predictions": {},
            "detections": [],
            "files": {},
        }

        candidate_detections = []
        for (x1, y1, x2, y2), score, label_id in zip(fused_boxes, fused_scores, fused_labels):
            class_name = self.display_names[int(label_id)]
            nutrition_profile = self.nutrition.lookup(class_name)

            candidate_detections.append({
                "class_name": class_name,
                "confidence": round(float(score), 4),
                "box_xyxy_norm": [round(float(v), 4) for v in (x1, y1, x2, y2)],
                "box_xyxy": [
                    round(float(x1) * w, 1), round(float(y1) * h, 1),
                    round(float(x2) * w, 1), round(float(y2) * h, 1),
                ],
                "nutrition_per_100g": nutrition_profile,
            })

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        classifier_top5, classifier_per_model = self.classifier_verifier.get_top5(pil_img)

        # Confirmation requires real agreement, not just top-5 membership.
        # A label that only ONE classifier sub-model happened to mention
        # (e.g. "fastfood" defaulting to "Pizza" on any round plated dish)
        # can still land in the aggregated top-5 purely on that one model's
        # score. Requiring >= MIN_MODELS_FOR_CONFIRMATION independent
        # sub-models to have surfaced the label makes "Confirmed" mean
        # actual cross-model agreement, not one classifier's own bias
        # lining up with a YOLO hallucination.
        strong_labels = {
            entry["label"] for entry in classifier_top5
            if entry.get("num_models", 1) >= MIN_MODELS_FOR_CONFIRMATION
        }

        confirmed_detections = [
            {**d, "source": "yolo_confirmed"} for d in candidate_detections
            if normalize_label(d["class_name"]) in strong_labels
        ]
        dropped_count = len(candidate_detections) - len(confirmed_detections)

        result["detections"] = confirmed_detections
        result["classifier_verification"] = {
            "top5": classifier_top5,
            "per_model": classifier_per_model,
            "candidates_before_verification": len(candidate_detections),
            "confirmed_after_verification": len(confirmed_detections),
            "dropped_unconfirmed": dropped_count,
        }

        # Fallback for when verification confirms nothing. Two distinct cases:
        #
        # 1. YOLO produced a specific candidate but the classifier ensemble
        #    couldn't corroborate it. This is very often a class-space
        #    mismatch (the classifier sub-models simply don't have that
        #    label at all, e.g. "idli"/"vada") rather than the detection
        #    being wrong. A specialist detector naming a specific dish is
        #    stronger evidence than an unrelated classifier top-1 guess, so
        #    prefer surfacing the YOLO candidate itself, flagged unverified.
        #
        # 2. YOLO produced nothing at all. Only then fall back to the
        #    classifier's own top-5, since it's the only signal available.
        confirmed_labels = {normalize_label(d["class_name"]) for d in confirmed_detections}
        unmatched_top5 = [
            entry for entry in classifier_top5
            if entry["label"] not in confirmed_labels
        ]

        if not confirmed_detections:
            if candidate_detections:
                best_candidate = max(candidate_detections, key=lambda d: d["confidence"])
                result["detections"] = [{**best_candidate, "source": "yolo_unverified"}]
            elif unmatched_top5:
                result["detections"] = [
                    {
                        "class_name": entry["display_label"],
                        "confidence": round(float(entry["score"]), 4),
                        "box_xyxy_norm": None,
                        "box_xyxy": None,
                        "nutrition_per_100g": self.nutrition.lookup(entry["display_label"]),
                        "source": "classifier_fallback",
                    }
                    for entry in unmatched_top5[:5]
                ]

        # `detections` above stays the clean, verified answer that downstream
        # logic (nutrition totals, etc.) should trust. For display purposes,
        # build a separate blended top-5 that also surfaces what the
        # classifier ensemble considered, even when a confirmed answer
        # already exists - e.g. showing "chole_bhature" and "chicken_curry"
        # as classifier alternatives alongside a confirmed "samosa".
        # Priority: confirmed YOLO > unconfirmed YOLO > classifier-only,
        # each tier ordered by its own confidence, deduped by label.
        top5_view = []
        seen_labels = set()

        for d in confirmed_detections:
            top5_view.append(d)
            seen_labels.add(normalize_label(d["class_name"]))

        for d in sorted(candidate_detections, key=lambda d: -d["confidence"]):
            label = normalize_label(d["class_name"])
            if label in seen_labels:
                continue
            top5_view.append({**d, "source": "yolo_unverified"})
            seen_labels.add(label)

        for entry in classifier_top5:
            if entry["label"] in seen_labels:
                continue
            # classifier_top5 scores are summed across sub-models and can
            # exceed 1.0 (e.g. two sub-models both picking a label as #1) -
            # cap for display so confidence reads as a sane percentage and
            # doesn't distort ordering below.
            capped_score = min(float(entry["score"]), 1.0)
            top5_view.append({
                "class_name": entry["display_label"],
                "confidence": round(capped_score, 4),
                "box_xyxy_norm": None,
                "box_xyxy": None,
                "nutrition_per_100g": self.nutrition.lookup(entry["display_label"]),
                "source": "classifier_suggestion",
            })
            seen_labels.add(entry["label"])

        top5_view.sort(key=lambda d: -d["confidence"])
        result["top5"] = top5_view[:5]

        montage_tiles = []

        for domain, boxes_px, scores, names in per_model_raw:
            result["raw_predictions"][domain] = [
                {
                    "class_name": name,
                    "confidence": round(score, 4),
                    "box_xyxy": [round(v, 1) for v in box],
                }
                for box, score, name in zip(boxes_px, scores, names)
            ]

            if save_visuals:
                vis = _draw_boxes(img, boxes_px, scores, names, color=(0, 140, 255))
                safe_domain = re.sub(r"[^a-zA-Z0-9_]", "_", domain)
                fname = f"raw_{safe_domain}.jpg"
                cv2.imwrite(str(run_dir / fname), vis)
                result["files"][f"raw_{safe_domain}"] = fname
                montage_tiles.append((domain, vis))

        if save_visuals and montage_tiles:
            build_montage(montage_tiles, run_dir / "montage.jpg")
            result["files"]["montage"] = "montage.jpg"

        return _to_native(result)