"""
AI-Based Multi-Language Movie Dubbing System
Main FastAPI application entry point
"""

import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import asyncio

from dubbing_pipeline import DubbingPipeline

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Movie Dubbing System",
    description="Translate and dub videos into multiple languages using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR   = Path("temp")

for d in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(exist_ok=True)

# ── In-memory job tracker ──────────────────────────────────────────────────────
jobs: dict[str, dict] = {}

# ── Request/Response models ────────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    job_id: str
    target_language: str          # e.g. "es", "fr", "de", "hi", "ja"
    whisper_model: Optional[str] = "base"   # tiny | base | small | medium | large

class JobStatus(BaseModel):
    job_id: str
    status: str                   # pending | processing | done | error
    progress: int                 # 0-100
    message: str
    detected_language: Optional[str] = None
    transcript: Optional[str] = None
    output_filename: Optional[str] = None

# ── Routes ──────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "AI Dubbing System API is running"}


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Step 1 – Accept video upload, save it, return a job_id.
    """
    # Validate file type
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {allowed}")

    job_id       = str(uuid.uuid4())
    saved_path   = UPLOAD_DIR / f"{job_id}{ext}"

    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "status": "uploaded",
        "progress": 0,
        "message": "Video uploaded successfully.",
        "input_path": str(saved_path),
        "detected_language": None,
        "transcript": None,
        "output_filename": None,
    }

    return {"job_id": job_id, "filename": file.filename, "message": "Upload successful"}


@app.post("/process-video")
async def process_video(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Step 2-9 – Kick off the full dubbing pipeline asynchronously.
    """
    job_id = req.job_id
    if job_id not in jobs:
        raise HTTPException(404, "Job ID not found. Please upload a video first.")

    job = jobs[job_id]
    if job["status"] == "processing":
        raise HTTPException(400, "Job is already processing.")

    job.update({"status": "processing", "progress": 5, "message": "Starting dubbing pipeline…"})

    background_tasks.add_task(
        run_pipeline,
        job_id,
        job["input_path"],
        req.target_language,
        req.whisper_model,
    )

    return {"job_id": job_id, "message": "Processing started"}


@app.get("/job-status/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Poll this endpoint to track progress."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=j["status"],
        progress=j["progress"],
        message=j["message"],
        detected_language=j.get("detected_language"),
        transcript=j.get("transcript"),
        output_filename=j.get("output_filename"),
    )


@app.get("/download-video/{job_id}")
def download_video(job_id: str):
    """Download the finished dubbed video."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, f"Video is not ready yet. Status: {job['status']}")

    output_path = OUTPUT_DIR / job["output_filename"]
    if not output_path.exists():
        raise HTTPException(500, "Output file missing on server.")

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=job["output_filename"],
    )


@app.get("/supported-languages")
def supported_languages():
    """Return the list of supported target languages."""
    return {
        "languages": [
            {"code": "af", "name": "Afrikaans"},
            {"code": "ar", "name": "Arabic"},
            {"code": "bg", "name": "Bulgarian"},
            {"code": "bn", "name": "Bengali"},
            {"code": "bs", "name": "Bosnian"},
            {"code": "ca", "name": "Catalan"},
            {"code": "cs", "name": "Czech"},
            {"code": "cy", "name": "Welsh"},
            {"code": "da", "name": "Danish"},
            {"code": "de", "name": "German"},
            {"code": "el", "name": "Greek"},
            {"code": "en", "name": "English"},
            {"code": "eo", "name": "Esperanto"},
            {"code": "es", "name": "Spanish"},
            {"code": "et", "name": "Estonian"},
            {"code": "fi", "name": "Finnish"},
            {"code": "fr", "name": "French"},
            {"code": "gu", "name": "Gujarati"},
            {"code": "hi", "name": "Hindi"},
            {"code": "hr", "name": "Croatian"},
            {"code": "hu", "name": "Hungarian"},
            {"code": "hy", "name": "Armenian"},
            {"code": "id", "name": "Indonesian"},
            {"code": "is", "name": "Icelandic"},
            {"code": "it", "name": "Italian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "jw", "name": "Javanese"},
            {"code": "km", "name": "Khmer"},
            {"code": "kn", "name": "Kannada"},
            {"code": "ko", "name": "Korean"},
            {"code": "la", "name": "Latin"},
            {"code": "lv", "name": "Latvian"},
            {"code": "mk", "name": "Macedonian"},
            {"code": "ml", "name": "Malayalam"},
            {"code": "mr", "name": "Marathi"},
            {"code": "my", "name": "Myanmar (Burmese)"},
            {"code": "ne", "name": "Nepali"},
            {"code": "nl", "name": "Dutch"},
            {"code": "no", "name": "Norwegian"},
            {"code": "pl", "name": "Polish"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ro", "name": "Romanian"},
            {"code": "ru", "name": "Russian"},
            {"code": "si", "name": "Sinhala"},
            {"code": "sk", "name": "Slovak"},
            {"code": "sq", "name": "Albanian"},
            {"code": "sr", "name": "Serbian"},
            {"code": "su", "name": "Sundanese"},
            {"code": "sv", "name": "Swedish"},
            {"code": "sw", "name": "Swahili"},
            {"code": "ta", "name": "Tamil"},
            {"code": "te", "name": "Telugu"},
            {"code": "th", "name": "Thai"},
            {"code": "tl", "name": "Filipino"},
            {"code": "tr", "name": "Turkish"},
            {"code": "uk", "name": "Ukrainian"},
            {"code": "ur", "name": "Urdu"},
            {"code": "vi", "name": "Vietnamese"},
            {"code": "zh-CN", "name": "Chinese (Simplified)"},
            {"code": "zh-TW", "name": "Chinese (Traditional)"},
            {"code": "zu", "name": "Zulu"},
        ]
    }


# ── Background task ────────────────────────────────────────────────────────────

def run_pipeline(job_id: str, input_path: str, target_language: str, whisper_model: str):
    """Runs the dubbing pipeline and updates job state throughout."""
    def update(progress: int, message: str, **kwargs):
        jobs[job_id].update({"progress": progress, "message": message, **kwargs})

    try:
        pipeline = DubbingPipeline(
            job_id=job_id,
            input_path=input_path,
            target_language=target_language,
            whisper_model=whisper_model,
            temp_dir=str(TEMP_DIR),
            output_dir=str(OUTPUT_DIR),
            update_callback=update,
        )
        output_filename = pipeline.run()
        jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "message": "Dubbing complete! Your video is ready to download.",
            "output_filename": output_filename,
        })
    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "progress": 0,
            "message": f"Error: {str(e)}",
        })
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
