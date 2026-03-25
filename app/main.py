"""
BeatCheck — FastAPI backend.

Serves the static frontend and exposes a single /analyse endpoint
that accepts an Apple Health CSV upload and returns a risk tier result.
"""

import os

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .pipeline import (
    parse_apple_health_export,
    parse_apple_health_xml,
    process_and_predict,
)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="BeatCheck", version="1.0.0")
app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Maximum 10 analyses per minute."},
    )

app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# ---------------------------------------------------------------------------
# CORS — driven by environment variable so it can be locked on Render
# Set ALLOWED_ORIGINS=https://your-app.onrender.com in Render env vars.
# Falls back to wildcard for local development.
# ---------------------------------------------------------------------------
_ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
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

# Maximum accepted upload size (backstop — client-side XML extraction already
# sends a small CSV, so real uploads are well under this limit)
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


def _error(status_code: int, detail: str) -> JSONResponse:
    """Return a JSON error response with the given status and message."""
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint for Render and UptimeRobot keep-alive."""
    return {"status": "ok"}


@app.get("/demo")
async def demo_result() -> dict:
    """
    Pre-baked sample result for live demo and server warm-up.
    Returns a realistic Intermediate-tier scenario — no file upload required.
    """
    return {
        "pct_flagged": 23.4,
        "total_windows": 256,
        "flagged_windows": 60,
        "risk_tier": "Intermediate",
        "days_analysed": 67,
    }


@app.api_route("/", methods=["GET", "HEAD"])
async def serve_index() -> FileResponse:
    """Serve the frontend HTML."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.post("/analyse", response_model=None)
@limiter.limit("10/minute")
async def analyse(request: Request, file: UploadFile = File(...)) -> dict | JSONResponse:
    """Accept Apple Health CSV or XML, run pipeline, return risk tier."""
    filename = (file.filename or "").lower()

    try:
        file_bytes = await file.read()
    except Exception as e:
        return _error(400, f"Failed to read uploaded file: {e}")

    # Log file size only — do not log user filenames
    print(f"[BeatCheck] Received {len(file_bytes):,} bytes")

    # Reject oversized uploads before parsing
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        return _error(
            400,
            f"File too large ({len(file_bytes) // (1024 * 1024)} MB). "
            "If uploading a full Apple Health export, use the XML file \u2014 "
            "it is processed in your browser before upload, sending only a small CSV.",
        )

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
