"""
BeatCheck — FastAPI backend.

Serves the static frontend and exposes a single /analyse endpoint
that accepts an Apple Health CSV upload and returns a risk tier result.
"""

import os

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import parse_apple_health_export, parse_apple_health_xml, process_and_predict

app = FastAPI(title="BeatCheck", version="1.0.0")

# CORS — allow all origins for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    """Serve the frontend HTML."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    """Accept Apple Health CSV or XML, run pipeline, return risk tier."""
    filename = (file.filename or "").lower()

    try:
        file_bytes = await file.read()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Failed to read uploaded file: {e}"},
        )

    print(f"[BeatCheck] Received {len(file_bytes):,} bytes from '{file.filename}'")

    if filename.endswith(".csv"):
        parser = parse_apple_health_export
    elif filename.endswith(".xml") or filename.endswith(".zip"):
        parser = parse_apple_health_xml
    else:
        return JSONResponse(
            status_code=400,
            content={"detail": "Unsupported file type. Please upload a .csv, .xml, or .zip file."},
        )

    try:
        df = parser(file_bytes)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    try:
        result = process_and_predict(df)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    return result
