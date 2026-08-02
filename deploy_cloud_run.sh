#!/usr/bin/env bash
set -e

# Configuration Variables
PROJECT_ID="gibud-f7cc9"
SERVICE_NAME="food-recognition-api"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "======================================================="
echo " Deploying Food Recognition API to Google Cloud Run"
echo " GCP Project: ${PROJECT_ID}"
echo " Service Name: ${SERVICE_NAME}"
echo " Region: ${REGION}"
echo "======================================================="

# Ensure gcloud CLI is configured to target project
gcloud config set project ${PROJECT_ID}

echo "[1/3] Enabling Google Container Registry / Artifact Registry and Cloud Run APIs..."
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com

echo "[2/3] Building container image using Google Cloud Build..."
gcloud builds submit --tag ${IMAGE_NAME} .

echo "[3/3] Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 80 \
  --timeout 300s \
  --set-env-vars "GCP_PROJECT=${PROJECT_ID},ENABLE_USDA_LIVE_API=true,SAVE_PREDICTION_IMAGES=false"

echo "======================================================="
echo " Deployment Complete!"
echo " Service URL:"
gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)'
echo "======================================================="
