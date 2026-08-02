"""
FastAPI Production REST Service for Multi-Model Food Detection & USDA Nutrition Estimation.

Designed for mobile (Flutter) and web client consumption. Deployable to Google Cloud Run.
"""

import logging
import shutil
import uuid
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import config
from pipeline.ensemble import FoodEnsemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ensemble: Optional[FoodEnsemble] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ensemble
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing Food Detection Ensemble from '%s'...", config.MODELS_ROOT)
    try:
        ensemble = FoodEnsemble(config.MODELS_ROOT)
        logger.info("Ensemble loaded successfully! Active domains: %s", ensemble.domain_names)
    except Exception as e:
        logger.error("Failed to load models during startup: %s", str(e))
        ensemble = None
    yield
    ensemble = None


app = FastAPI(
    title="Food Recognition & USDA Nutrition Estimation API",
    description=(
        "Production-ready multi-model YOLO ensemble service with Weighted Boxes Fusion "
        "and USDA FoodData Central integration tailored for Flutter applications."
    ),
    version=config.API_VERSION,
    lifespan=lifespan,
)

# CORS Configuration suitable for Flutter Web/Mobile and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs static folder if local visual saving is enabled
if config.OUTPUT_DIR.is_dir():
    app.mount("/outputs", StaticFiles(directory=str(config.OUTPUT_DIR)), name="outputs")

if Path("static").is_dir():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic Schemas for OpenAPI / Swagger & Flutter Type Safety
class ImageDimensions(BaseModel):
    width: int
    height: int


class NutritionProfile(BaseModel):
    calories_kcal: Optional[float] = Field(None, description="Energy in kcal")
    protein_g: Optional[float] = Field(None, description="Protein in grams")
    carbs_g: Optional[float] = Field(None, description="Carbohydrates in grams")
    fat_g: Optional[float] = Field(None, description="Total Fat in grams")
    fiber_g: Optional[float] = Field(None, description="Dietary Fiber in grams")
    sugars_g: Optional[float] = Field(None, description="Total Sugars in grams")
    sodium_mg: Optional[float] = Field(None, description="Sodium in mg")
    potassium_mg: Optional[float] = Field(None, description="Potassium in mg")
    calcium_mg: Optional[float] = Field(None, description="Calcium in mg")
    iron_mg: Optional[float] = Field(None, description="Iron in mg")
    vitamin_a_iu: Optional[float] = Field(None, description="Vitamin A in IU")
    vitamin_c_mg: Optional[float] = Field(None, description="Vitamin C in mg")
    vitamin_d_iu: Optional[float] = Field(None, description="Vitamin D in IU")
    serving_size: Optional[float] = Field(100.0, description="Serving size amount")
    serving_size_unit: Optional[str] = Field("g", description="Serving size unit")
    source: Optional[str] = Field(None, description="Nutritional DB source")
    matched_name: Optional[str] = Field(None, description="Matched database item name")
    match_type: Optional[str] = Field(None, description="Matching algorithm used")


class DetectedFoodItem(BaseModel):
    class_name: str
    confidence: float
    box_xyxy_norm: Optional[List[float]] = Field(None, description="[x_min, y_min, x_max, y_max] normalized 0.0-1.0")
    box_xyxy: Optional[List[float]] = Field(None, description="[x_min, y_min, x_max, y_max] in pixels")
    source: str = Field("yolo_confirmed", description="Detection source tag")
    nutrition_per_100g: Optional[NutritionProfile] = None


class TotalNutritionalSummary(BaseModel):
    total_items_detected: int
    total_calories_kcal: float
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    total_fiber_g: float
    total_sugars_g: float
    total_sodium_mg: float


class DetectionResponse(BaseModel):
    success: bool
    run_id: str
    image_size: ImageDimensions
    detections: List[DetectedFoodItem]
    top5: List[DetectedFoodItem]
    nutrition_summary: TotalNutritionalSummary
    urls: Dict[str, str] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    domains: List[str]
    usda_api_enabled: bool


class VersionResponse(BaseModel):
    version: str
    environment: str
    project_id: str


@app.get("/", include_in_schema=False)
def root():
    if Path("static/index.html").is_file():
        return FileResponse("static/index.html")
    return {"message": "Food Recognition API is running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    if ensemble is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded yet or failed to initialize",
        )
    return {
        "status": "healthy",
        "models_loaded": True,
        "domains": ensemble.domain_names,
        "usda_api_enabled": config.ENABLE_USDA_LIVE_API,
    }


@app.get("/version", response_model=VersionResponse)
def get_version():
    return {
        "version": config.API_VERSION,
        "environment": os.getenv("K_SERVICE", "local-dev"),
        "project_id": os.getenv("GCP_PROJECT", "gibud-f7cc9"),
    }


@app.get("/models")
def models_info():
    if ensemble is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded",
        )
    return ensemble.info()


@app.post("/predict", response_model=DetectionResponse)
async def predict(
    file: UploadFile = File(...),
    save_visuals: bool = Query(False, description="Set True if image montage/annotated output should be saved to disk"),
):
    if ensemble is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are initializing or unavailable",
        )

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file.content_type}. Please upload JPEG, PNG, or WebP.",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {config.MAX_UPLOAD_MB} MB.",
        )

    img_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image data. The image file might be corrupted.",
        )

    run_id = uuid.uuid4().hex[:12]
    run_dir = config.OUTPUT_DIR / run_id
    if save_visuals:
        run_dir.mkdir(parents=True, exist_ok=True)

    result = ensemble.predict(img, run_dir=run_dir, save_visuals=save_visuals)

    urls = {}
    if save_visuals and "files" in result:
        urls = {key: f"/outputs/{run_id}/{fname}" for key, fname in result["files"].items()}

    # Compute nutritional totals across all detected food items
    total_cal = 0.0
    total_prot = 0.0
    total_carbs = 0.0
    total_fat = 0.0
    total_fiber = 0.0
    total_sugars = 0.0
    total_sodium = 0.0

    for d in result.get("detections", []):
        nutr = d.get("nutrition_per_100g") or {}
        total_cal += float(nutr.get("calories_kcal") or 0.0)
        total_prot += float(nutr.get("protein_g") or 0.0)
        total_carbs += float(nutr.get("carbs_g") or 0.0)
        total_fat += float(nutr.get("fat_g") or 0.0)
        total_fiber += float(nutr.get("fiber_g") or 0.0)
        total_sugars += float(nutr.get("sugars_g") or 0.0)
        total_sodium += float(nutr.get("sodium_mg") or 0.0)

    summary = {
        "total_items_detected": len(result.get("detections", [])),
        "total_calories_kcal": round(total_cal, 2),
        "total_protein_g": round(total_prot, 2),
        "total_carbs_g": round(total_carbs, 2),
        "total_fat_g": round(total_fat, 2),
        "total_fiber_g": round(total_fiber, 2),
        "total_sugars_g": round(total_sugars, 2),
        "total_sodium_mg": round(total_sodium, 2),
    }

    return {
        "success": True,
        "run_id": run_id,
        "image_size": result["image_size"],
        "detections": result.get("detections", []),
        "top5": result.get("top5", []),
        "nutrition_summary": summary,
        "urls": urls,
    }


@app.delete("/outputs/{run_id}")
def cleanup_run(run_id: str):
    run_dir = config.OUTPUT_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id output directory not found")
    shutil.rmtree(run_dir)
    return {"deleted": run_id}