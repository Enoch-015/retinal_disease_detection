from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from .inference import InferenceError, InvalidImageError, ModelService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
service = ModelService(model_dir=PROJECT_ROOT / "models")


def allowed_origins() -> list[str]:
    configured = os.getenv("ALLOWED_ORIGINS", "")
    if configured.strip():
        return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    return ["http://localhost:5173", "http://127.0.0.1:5173"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    yield


app = FastAPI(title="OCT Image Inference API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", responses={503: {"description": "Model unavailable"}})
async def health(response: Response) -> dict:
    if not service.loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return service.health()


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a valid image file.")

    image_bytes = await image.read()

    try:
        return await run_in_threadpool(service.predict, image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
