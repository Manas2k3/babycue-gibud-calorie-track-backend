"""
Central configuration for the food detection ensemble API.
All values can be overridden with environment variables of the same name,
e.g. on Windows (PowerShell): $env:FINAL_CONF_THRESH="0.4"
     on macOS/Linux:          export FINAL_CONF_THRESH=0.4
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Parent folder containing one subfolder per trained model, e.g.:
#   models/
#     indian fast food/xxx.pt
#     indian_food/xxx.pt
#     indianfoodnet-30/xxx.pt
#     soft drinks/xxx.pt
#     western food/xxx.pt
MODELS_ROOT = Path(os.getenv("MODELS_ROOT", BASE_DIR / "models"))

# Where annotated images (raw per-model, montage) get written.
# Served back to clients via the /outputs static mount.
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "outputs"))

# Per-domain weight overrides (by folder name). Anything not listed here
# defaults to 1.0. Tune these once you know which domains are more reliable,
# e.g. {"soft drinks": 1.3, "indianfoodnet-30": 0.7}
DOMAIN_WEIGHT_OVERRIDES: dict[str, float] = {}

# Per-model confidence threshold BEFORE fusion. Kept low so WBF has
# candidates to average from every model; real filtering happens after fusion.
PER_MODEL_CONF_THRESH = float(os.getenv("PER_MODEL_CONF_THRESH", "0.15"))

# Weighted Boxes Fusion params
WBF_IOU_THR = float(os.getenv("WBF_IOU_THR", "0.55"))       # overlap threshold to fuse boxes together
WBF_SKIP_BOX_THR = float(os.getenv("WBF_SKIP_BOX_THR", "0.0"))  # 0 = keep all, filter after fusion

# Final confidence threshold applied AFTER fusion (your real detection threshold)
FINAL_CONF_THRESH = float(os.getenv("FINAL_CONF_THRESH", "0.35"))

# Minimum number of the 8 classifier sub-models that must independently
# surface a label (see classifier_verification.py's "num_models" field)
# for a YOLO detection matching that label to count as "Confirmed" rather
# than just top-5 membership. 1 = old behavior (any single sub-model's
# opinion is enough); 2+ requires real cross-model agreement, which avoids
# a YOLO hallucination getting rubber-stamped just because it happens to
# match one classifier's own bias (e.g. the fastfood model defaulting to
# "Pizza" on round plated dishes in general).
MIN_MODELS_FOR_CONFIRMATION = int(os.getenv("MIN_MODELS_FOR_CONFIRMATION", "2"))

# Max upload size in megabytes
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))

# CORS - which origins can call this API from a browser. "*" = allow all
# (fine for local dev / a demo; tighten this for production).
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")

# Classes to always exclude from results, even if a model detects them.
# This matters because some of your domain folders currently hold the
# STOCK pretrained COCO checkpoint (not yet fine-tuned on your food data),
# so it will happily detect "person", "car", "dog", etc. This list filters
# those non-food COCO classes out of the unified class space entirely, so
# they can never appear in raw predictions - regardless of which model
# produced them. Food-relevant COCO classes (banana, pizza, donut,
# sandwich, etc.) are deliberately NOT excluded, since they're real food
# items a model could legitimately detect.
#
# "carbonated_soft_drinks" is disabled here by request - it's the
# normalized form of the "carbonated-soft-drinks" class from the
# "soft drinks" domain model (normalize_name() collapses hyphens/spaces
# to underscores), so this excludes it everywhere: raw_predictions,
# detections, and top5.
NON_FOOD_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports_ball", "kite", "baseball_bat", "baseball_glove", "skateboard",
    "surfboard", "tennis_racket", "bottle", "wine_glass", "cup", "fork",
    "knife", "spoon", "bowl", "chair", "couch", "potted_plant", "bed",
    "dining_table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell_phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy_bear", "hair_drier",
    "toothbrush",
    "carbonated_soft_drinks",
}

# TEMPORARY: domains (folder names) that are not yet properly fine-tuned
# (still holding a stock/placeholder checkpoint). Weighted Boxes Fusion
# penalizes any detection that only ONE model agrees on, scaling its
# confidence down by roughly (models_that_detected_it / total_models). With
# 3 of 5 domains non-functional, a genuinely correct 92% detection from the
# one working model gets crushed down to ~18% and silently dropped. Listing
# a domain here excludes it from the FUSION MATH ONLY - its raw predictions
# still show up in the montage/raw_predictions output so you can keep an
# eye on it. Remove a domain from this set once you've swapped in its real
# trained weights.
DISABLED_DOMAINS: set[str] = {
    "indian_food",
    "indianfoodnet-30",
    "western food",
}

# Nutrition lookup tables, in priority order (first match wins). Each entry
# is a CSV with at least a "name" column plus calorie/macro columns - see
# nutrition.py's COLUMN_MAP if your column headers differ from the default
# (calories_kcal, protein_g, carbs_g, fat_g, fiber_g). Put your INDB/Anuvaad
# table first since it's tuned for Indian dishes, and use USDA FoodData
# Central as the fallback for anything it doesn't cover.
NUTRITION_DB_DIR = Path(os.getenv("NUTRITION_DB_DIR", BASE_DIR / "data" / "nutrition"))
NUTRITION_DB_PATHS = [
    Path(p) for p in os.getenv(
        "NUTRITION_DB_PATHS",
        f"{NUTRITION_DB_DIR / 'indb_anuvaad_2024.csv'},{NUTRITION_DB_DIR / 'usda_fooddata_central.csv'}",
    ).split(",")
]

# USDA FoodData Central REST API Configuration
USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")
ENABLE_USDA_LIVE_API = os.getenv("ENABLE_USDA_LIVE_API", "true").lower() in ("true", "1", "yes")

# Service Deployment & Latency Optimization Settings
PORT = int(os.getenv("PORT", "8080"))
API_VERSION = "1.1.0"
SAVE_PREDICTION_IMAGES = os.getenv("SAVE_PREDICTION_IMAGES", "false").lower() in ("true", "1", "yes")