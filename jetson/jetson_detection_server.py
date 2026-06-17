#!/usr/bin/env python3
"""
jetson_detection_server.py — HTTP object-detection offload server.

Runs on a GPU workstation (e.g. lkk-alienware51 with GTX 1080 Ti) and lets many
Jetson devices offload detection over HTTP — no SSH, no per-device keys. It
reuses the SAME detector classes as jetson_object_detection_toolkit.py, so
results are identical to running locally.

Run:
    DETECT_API_KEY=sjsudetect uvicorn jetson_detection_server:app --host 0.0.0.0 --port 8000
    # or: python3 jetson_detection_server.py   (uses uvicorn programmatically)

API (OpenAI-style bearer auth when DETECT_API_KEY is set):
    GET  /health           -> status + GPU info + loaded models
    GET  /v1/models        -> supported detector model types
    POST /detect           -> JSON {model, image_b64, confidence, iou, prompts?, return_image?}
                              returns {num_objects, inference_time_ms, detections[], image_b64?}
"""
import os
import base64
import time
from typing import Optional, List

import numpy as np
import cv2
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Reuse the exact detectors/visualization from the toolkit (same directory).
from jetson_object_detection_toolkit import ObjectDetectionToolkit
import torch

API_KEY = os.environ.get("DETECT_API_KEY", "")  # empty = no auth required
SUPPORTED = ['faster-rcnn', 'maskrcnn', 'yolo', 'owl-vit', 'grounding-dino',
             'detr', 'detr-resnet-101', 'conditional-detr', 'rt-detr']

app = FastAPI(title="Jetson Offload Detection Server", version="1.0")
_toolkits = {}  # model_type -> ObjectDetectionToolkit (lazy-loaded + cached)


def _get_toolkit(model_type: str) -> ObjectDetectionToolkit:
    if model_type not in SUPPORTED:
        raise HTTPException(400, f"unsupported model '{model_type}'; choose from {SUPPORTED}")
    if model_type not in _toolkits:
        kwargs = {}
        if model_type == 'yolo':
            kwargs['model_path'] = 'yolov8n.pt'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _toolkits[model_type] = ObjectDetectionToolkit(model_type, device, **kwargs)
    return _toolkits[model_type]


def _check_auth(authorization: Optional[str]):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid or missing API key (Authorization: Bearer <key>)")


class DetectRequest(BaseModel):
    model: str = "yolo"
    image_b64: str
    confidence: float = 0.25
    iou: float = 0.45
    prompts: Optional[str] = None         # for owl-vit / grounding-dino (comma-separated)
    return_image: bool = True             # return the annotated image as base64 JPEG


@app.get("/health")
def health():
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] \
        if torch.cuda.is_available() else []
    return {"status": "ok", "cuda": torch.cuda.is_available(), "gpus": gpus,
            "loaded_models": list(_toolkits.keys()), "auth_required": bool(API_KEY)}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": m} for m in SUPPORTED]}


@app.post("/detect")
def detect(req: DetectRequest, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)

    # decode the image
    try:
        buf = np.frombuffer(base64.b64decode(req.image_b64), np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("could not decode image")
    except Exception as e:
        raise HTTPException(400, f"bad image_b64: {e}")

    toolkit = _get_toolkit(req.model)

    # map params to the per-model detect() signature (mirrors the CLI)
    dkw = {}
    if req.model == 'yolo':
        dkw = {'conf_threshold': req.confidence, 'iou_threshold': req.iou}
    elif req.model == 'owl-vit':
        if not req.prompts:
            raise HTTPException(400, "owl-vit requires 'prompts'")
        dkw = {'confidence_threshold': req.confidence,
               'text_prompts': [p.strip() for p in req.prompts.split(',')]}
    elif req.model == 'grounding-dino':
        if not req.prompts:
            raise HTTPException(400, "grounding-dino requires 'prompts'")
        dkw = {'box_threshold': req.confidence, 'text_prompt': req.prompts}
    else:  # faster-rcnn, maskrcnn, detr family
        dkw = {'confidence_threshold': req.confidence}

    t0 = time.time()
    results = toolkit.detect(image, **dkw)
    server_ms = (time.time() - t0) * 1000

    resp = {
        "model": req.model,
        "num_objects": int(len(results['boxes'])),
        "inference_time_ms": float(results.get('inference_time', server_ms)),
        "detections": [
            {"class": cn, "score": float(s),
             "box": [float(x) for x in box]}
            for box, s, cn in zip(results['boxes'], results['scores'], results['class_names'])
        ],
    }
    if req.return_image:
        annotated = toolkit.visualize_results(image, results)
        ok, jpg = cv2.imencode('.jpg', annotated)
        if ok:
            resp["image_b64"] = base64.b64encode(jpg.tobytes()).decode()
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DETECT_PORT", "8000")))
