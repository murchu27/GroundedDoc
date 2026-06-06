- Cursor finished initial plan
- Created GCP project (Name: GroundedDoc, ID: zeta-antenna-498616-h0)
- Enabled "Secret Manager API" (secretmanager.googleapis.com)
- Created Gemini API Key on Google AI Studio (https://aistudio.google.com/app/api-keys)
- Created a Secret on Secret Manager for the Gemini API Key (ID: grounded-doc-gemini-key)
    - NOTE: Also stored in Bitwarden under Google (murchu27@yorku.ca)

- Installed `gcloud` as per the [Google Cloud SDK docs](https://docs.cloud.google.com/sdk/docs/install-sdk).
```bash
cd ~/bin
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
```

- Cursor updated the deploy script (`deploy_cloud_run.sh`), as the `gcloud builds submit...` command was formatted wrong
- Authenticated `gcloud` with `gcloud auth login`.

- Attempted to deploy:
```bash
export GCP_PROJECT_ID=zeta-antenna-498616-h0
./scripts/deploy_cloud_run.sh
```

- `deploy_cloud_run.sh` failed, seems like a race condition; tried to deploy before APIs were enabled.
- Cursor updated `deploy_cloud_run.sh` to enable APIs explicitly, and to repeatedly try the `gcloud builds submit` step until the API is finished enabling
- Cursor also updated one of the agents, which had a `NameError` due to a missing import.

- Ran `deploy_cloud_run.sh` again; this time, build succeeded, but deploying the Cloud Run service did not.
- Cursor analysed; the `gcloud run deploy` step was missing a `--project` argument, and also the default compute
  service account (which the Cloud Run service runs as) didn't have access to the secret.
- These gaps were addressed in `deploy_cloud_run.sh`, but I won't run that again since it would build the full container again

- Instead, giving the required permissions and running the deploy step only with:
```bash
export GCP_PROJECT_ID=zeta-antenna-498616-h0

PROJECT_NUMBER=$(gcloud projects describe "$GCP_PROJECT_ID" --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding grounded-doc-gemini-key \
  --project="$GCP_PROJECT_ID" \
  --member "serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role "roles/secretmanager.secretAccessor"

gcloud run deploy grounded-doc-agent \
  --image "gcr.io/${GCP_PROJECT_ID}/grounded-doc-agent:latest" \
  --project "$GCP_PROJECT_ID" \
  --platform managed \
  --region northamerica-northeast2 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --set-secrets "GOOGLE_API_KEY=grounded-doc-gemini-key:latest" \
  --set-env-vars "GROUNDED_REQUIRE_API_KEY=true,MLFLOW_TRACKING_URI=/tmp/mlruns"
```

- This deployed successfully, and the application URL is accessible (though it gives an application-level 404)