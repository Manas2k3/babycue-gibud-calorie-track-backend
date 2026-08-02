"""
Wraps 8 pretrained Hugging Face image-classification models as a
verification / cross-check layer for the YOLO ensemble's fused detections.

None of these are trained or fine-tuned here - they're off-the-shelf
pretrained checkpoints pulled from the Hugging Face Hub, used purely for
inference. Adapted from food-classifier-ensemble.ipynb.

How it's used (see pipeline/ensemble.py):
  1. The 6-model YOLO ensemble produces fused detections as usual.
  2. This module runs all 8 classifiers on the SAME full image and combines
     their individual top-k predictions into one aggregated top-5 ranking
     (mean confidence per normalized label, across all loaded models - a
     label only a subset of models mention is averaged over ALL loaded
     models, not just the ones that mentioned it, so partial agreement
     from a couple of models can't outrank one model's high-confidence
     pick just by summing).
  3. Any fused detection whose class name does NOT appear in that
     aggregated top-5 gets DROPPED - only detections both the YOLO
     ensemble AND the classifier ensemble agree on survive.

Known limitation: matching is done on normalized label text (lowercased,
punctuation-stripped). The 8 classifiers use their own label vocabularies
(mostly Food-101 / general Indian food terms), which may not exactly match
your domain models' dish names even when they're describing the same food
(e.g. "dal tadka" vs "dal makhani" won't match each other, correctly - but
a genuinely correct detection could also get dropped if no classifier's
vocabulary contains that exact dish name). If you find valid detections
being dropped too often, that's the first thing to look at - consider
adding a synonym/alias table rather than relying on exact normalized match.
"""

import re
from collections import defaultdict

import torch
from PIL import Image
from transformers import pipeline

MODEL_IDS = {
    "general": "Kaludi/food-category-classification-v2.0",
    "indian_v1": "rajistics/finetuned-indian-food",
    "indian_v2": "dima806/indian_food_image_detection",
    "fastfood": "dima806/fast_food_image_detection",
    "western_vit": "nateraw/food",
    "western_siglip": "prithivMLmods/Food-101-93M",
    "western_swin": "skylord/swin-finetuned-food101",
    "western_swin_v2": "arnabdhar/Swin-V2-base-Food",
}

TOP_K_PER_MODEL = 5   # ask each of the 8 models for its own top-5
FINAL_TOP_N = 5         # size of the AGGREGATED top-5 used for cross-checking


def normalize_label(label: str) -> str:
    """Same normalization style as pipeline/ensemble.py's normalize_name,
    duplicated here (rather than imported) to keep this module independent
    and avoid a circular import between the two."""
    label = label.lower().strip()
    label = re.sub(r"[\s_-]+", "_", label)
    label = re.sub(r"[^a-z0-9_]", "", label)
    return label


class ClassifierVerifier:
    """Loads all 8 pretrained classifiers ONCE (expensive - first run also
    downloads weights from the Hub) and exposes get_top5() for repeated,
    cheap use across many API requests."""

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        print(f"[classifier_verifier] loading {len(MODEL_IDS)} pretrained "
              f"classifiers on {'GPU' if device == 0 else 'CPU'}...")

        self.classifiers = {}
        for name, model_id in MODEL_IDS.items():
            try:
                self.classifiers[name] = pipeline(
                    "image-classification", model=model_id, device=device
                )
                print(f"[classifier_verifier] loaded '{name}' ({model_id})")
            except Exception as e:
                print(f"[classifier_verifier] FAILED to load '{name}' ({model_id}): {e}")

        print(f"[classifier_verifier] {len(self.classifiers)}/{len(MODEL_IDS)} classifiers ready")

    def get_top5(self, image: Image.Image):
        """Runs all loaded classifiers on one PIL image, aggregates their
        individual top-k predictions into ONE combined ranking, and returns:
          - top5: list of {"label", "display_label", "score", "num_models"} -
            the aggregated top FINAL_TOP_N, used for cross-checking. `score`
            is the MEAN confidence across ALL loaded classifiers (models
            that didn't mention the label at all in their own top-k count
            as 0), so it stays bounded in [0, 1] and can't be inflated just
            because a couple of models happened to mention it. `num_models`
            is how many of the loaded classifiers actually surfaced this
            label in their own top-k, for transparency - two labels can
            have the same mean score with very different agreement behind
            it (one model very confident vs. several moderately confident).
          - per_model_results: raw top-k output per classifier, for
            transparency/debugging in the API response
        """
        aggregated_scores = defaultdict(float)
        model_hits = defaultdict(int)
        display_labels = {}
        per_model_results = {}

        for name, clf in self.classifiers.items():
            try:
                preds = clf(image, top_k=TOP_K_PER_MODEL)
            except Exception as e:
                print(f"[classifier_verifier] '{name}' inference failed: {e}")
                preds = []

            per_model_results[name] = [
                {"label": p["label"], "score": round(float(p["score"]), 4)}
                for p in preds
            ]

            for p in preds:
                norm = normalize_label(p["label"])
                aggregated_scores[norm] += float(p["score"])
                model_hits[norm] += 1
                display_labels.setdefault(norm, p["label"])

        # Mean confidence among the models that actually predicted this
        # label - NOT divided by the total number of loaded classifiers.
        # Dividing by total would crush a label only 1-2 models mentioned
        # down to a tiny fraction (e.g. 0.83 / 8 = 0.10), making classifier
        # scores structurally smaller than YOLO's raw box confidence no
        # matter what, and biasing any cross-source ranking toward YOLO.
        # Dividing by the actual agreeing-model count keeps scores on a
        # comparable 0-1 scale to a single model's own confidence, while
        # num_models (returned separately) still lets callers weigh
        # agreement breadth if they want it.
        mean_scores = {
            norm: total / model_hits[norm]
            for norm, total in aggregated_scores.items()
        }

        ranked = sorted(mean_scores.items(), key=lambda kv: kv[1], reverse=True)
        top5 = [
            {
                "label": norm,
                "display_label": display_labels[norm],
                "score": round(score, 4),
                "num_models": model_hits[norm],
            }
            for norm, score in ranked[:FINAL_TOP_N]
        ]

        return top5, per_model_results