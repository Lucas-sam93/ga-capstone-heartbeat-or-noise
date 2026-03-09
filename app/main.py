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

from .pipeline import (
    parse_apple_health_export,
    parse_apple_health_xml,
    process_and_predict,
)

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

# File extension to parser mapping
_PARSERS = {
    ".csv": parse_apple_health_export,
    ".xml": parse_apple_health_xml,
}


def _error(status_code: int, detail: str) -> JSONResponse:
    """Return a JSON error response with the given status and message."""
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint for Render."""
    return {"status": "ok"}


@app.api_route("/", methods=["GET", "HEAD"])
async def serve_index() -> FileResponse:
    """Serve the frontend HTML."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.post("/analyse")
async def analyse(file: UploadFile = File(...)) -> dict | JSONResponse:
    """Accept Apple Health CSV or XML, run pipeline, return risk tier."""
    filename = (file.filename or "").lower()

    try:
        file_bytes = await file.read()
    except Exception as e:
        return _error(400, f"Failed to read uploaded file: {e}")

    print(f"[BeatCheck] Received {len(file_bytes):,} bytes from '{file.filename}'")

    # Select parser by file extension
    parser = next(
        (p for ext, p in _PARSERS.items() if filename.endswith(ext)),
        None,
    )
    if parser is None:
        return _error(400, "Unsupported file type. Please upload a .csv or .xml file.")

    try:
        df = parser(file_bytes)
    except ValueError as e:
        return _error(400, str(e))
    except Exception as e:
        return _error(400, f"Could not parse the uploaded file: {e}")

    try:
        result = process_and_predict(df)
    except ValueError as e:
        return _error(400, str(e))
    except Exception as e:
        return _error(500, f"Analysis failed unexpectedly: {e}")

    return result
