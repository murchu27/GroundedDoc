#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-northamerica-northeast2}"
SERVICE_NAME="${SERVICE_NAME:-grounded-doc-agent}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

# Enable required APIs up front so the first deploy does not fail while
# cloudbuild.googleapis.com is still propagating after an interactive enable.
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  --project "${PROJECT_ID}" \
  --quiet

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for secret in grounded-doc-gemini-key grounded-doc-api-key; do
  gcloud secrets add-iam-policy-binding "${secret}" \
    --project "${PROJECT_ID}" \
    --member "serviceAccount:${RUNTIME_SA}" \
    --role "roles/secretmanager.secretAccessor" \
    --quiet >/dev/null
done

submit_build() {
  gcloud builds submit \
    --config infra/cloudbuild.yaml \
    --substitutions "_SERVICE_NAME=${SERVICE_NAME},_IMAGE_TAG=latest" \
    --project "${PROJECT_ID}" \
    .
}

if ! submit_build; then
  echo "Build submit failed; waiting 30s for Cloud Build API propagation and retrying once..."
  sleep 30
  submit_build
fi

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --project "${PROJECT_ID}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --set-secrets "GOOGLE_API_KEY=grounded-doc-gemini-key:latest,GROUNDED_API_KEY=grounded-doc-api-key:latest" \
  --set-env-vars "GROUNDED_REQUIRE_API_KEY=true,MLFLOW_TRACKING_URI=/tmp/mlruns,GROUNDED_CORS_ORIGINS=*"

echo "Deployed ${SERVICE_NAME} to Cloud Run"
