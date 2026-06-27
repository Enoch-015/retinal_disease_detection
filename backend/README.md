# OCT Inference API

FastAPI service for image-only OCT inference using a dynamically quantized LiteRT
model.

## Local Development

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
```

The API expects `models/oct_resnet50.tflite` at the repository root.

## Endpoints

- `GET /api/health`
- `POST /api/predict` with multipart form field `image`

The health endpoint returns `200` only after the model is loaded and `503` when
inference is unavailable. Uploaded images are resized to 224x224 RGB before
inference.

## Model Conversion

The original Keras files are intentionally ignored because each copy is 204 MB.
Rebuild and verify the tracked 8-bit dynamic-range TFLite artifact with:

```bash
cd backend
python tools/convert_model.py ../models/oct_resnet50.keras ../models/oct_resnet50.tflite
python tools/verify_model.py ../models/oct_resnet50.keras ../models/oct_resnet50.tflite
```

Pass OCT image paths after the two model arguments to run parity checks on real
samples. With no image paths, the verifier uses deterministic synthetic inputs.

## Render

The root-level `render.yaml` creates a single free web service. Create a Render
Blueprint from the repository and enter the Vercel production origin when
prompted for `ALLOWED_ORIGINS`:

```text
https://your-project.vercel.app
```

Multiple exact origins can be supplied as a comma-separated list. Do not include
paths. Render uses `/api/health` as its readiness check and starts one Uvicorn
worker to stay within the free instance's memory allowance.

After Render assigns the service URL, configure this Vercel environment variable
and redeploy the frontend:

```text
VITE_API_BASE_URL=https://your-render-service.onrender.com
```
