"""
Video Composer API - Auto-Scaling Overlay Support
==================================================
POST /compose → Start async video composition job
GET /status/{job_id} → Check job progress
GET /download/{job_id} → Download completed video

Features:
- Auto-scales overlay to match base video resolution
- Async processing for long videos (2+ hours)
- Progress tracking with percentage
- Automatic cleanup after 48 hours
"""

import os
import subprocess
import tempfile
import logging
import uuid
import threading
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import shutil

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Video Composer API",
    version="2.0.0",
    description="Async video composition with auto-scaling overlay support"
)

# In-memory job storage
jobs: Dict[str, dict] = {}
jobs_lock = threading.Lock()

# Cleanup configuration
MAX_JOB_AGE_HOURS = 48

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ComposeRequest(BaseModel):
    """Request model for video composition"""
    base_video_url: str = Field(
        ..., 
        description="Base video URL (MP4) - determines output resolution",
        example="https://cdn.example.com/base_video.mp4"
    )
    overlay_video_url: str = Field(
        ..., 
        description="Overlay video URL (MP4) - will be auto-scaled to match base",
        example="https://cdn.example.com/overlay.mp4"
    )
    background_audio_url: str = Field(
        ..., 
        description="Background audio URL (MP3)",
        example="https://cdn.example.com/ambient.mp3"
    )
    overlay_opacity: float = Field(
        default=0.25, 
        ge=0.0, 
        le=1.0, 
        description="Overlay opacity (0.0=invisible, 1.0=fully visible). Recommended: 0.15-0.35 for subtle effects"
    )
    background_volume: float = Field(
        default=0.15, 
        ge=0.0, 
        le=1.0, 
        description="Background audio volume (0.0=mute, 1.0=full volume)"
    )
    base_video_volume: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Base video audio volume (if base has audio)"
    )

    class Config:
        schema_extra = {
            "example": {
                "base_video_url": "https://cdn.example.com/sleep_story.mp4",
                "overlay_video_url": "https://cdn.example.com/rain_overlay.mp4",
                "background_audio_url": "https://cdn.example.com/ocean_ambient.mp3",
                "overlay_opacity": 0.25,
                "background_volume": 0.15,
                "base_video_volume": 1.0
            }
        }


class JobResponse(BaseModel):
    """Response after creating a job"""
    job_id: str
    status: str
    message: str
    status_url: str
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "abc123-def456-ghi789",
                "status": "pending",
                "message": "Job created successfully",
                "status_url": "/status/abc123-def456-ghi789"
            }
        }


class StatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: str  # pending, processing, complete, failed
    progress: int  # 0-100
    message: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    download_url: Optional[str] = None
    file_size_mb: Optional[float] = None
    video_info: Optional[Dict] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_video_info(video_path: str) -> Dict:
    """
    Extract video metadata using ffprobe
    
    Returns:
        dict: {width, height, duration, has_audio}
    """
    try:
        # Get video stream info
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise Exception(f"ffprobe error: {result.stderr}")
        
        data = json.loads(result.stdout)
        stream = data.get('streams', [{}])[0]
        
        width = stream.get('width', 1920)
        height = stream.get('height', 1080)
        duration = float(stream.get('duration', 0))
        
        # Check for audio stream
        cmd_audio = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result_audio = subprocess.run(cmd_audio, capture_output=True, text=True, timeout=10)
        has_audio = result_audio.stdout.strip() == 'audio'
        
        return {
            'width': width,
            'height': height,
            'duration': duration,
            'has_audio': has_audio
        }
        
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {
            'width': 1920,
            'height': 1080,
            'duration': 0,
            'has_audio': False
        }


def download_file(url: str, dest_path: str, job_id: str, file_type: str) -> str:
    """
    Download file with progress logging
    
    Args:
        url: Source URL
        dest_path: Destination path
        job_id: Job ID for logging
        file_type: "base_video", "overlay", or "audio"
    """
    logger.info(f"[{job_id}] Downloading {file_type}: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=900)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):  # 2MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size and downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                        progress = int((downloaded / total_size) * 100)
                        logger.info(f"[{job_id}] {file_type} download: {progress}%")
        
        file_size = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"[{job_id}] Downloaded {file_type}: {file_size:.1f}MB")
        return dest_path
        
    except Exception as e:
        logger.error(f"[{job_id}] Download failed for {file_type}: {e}")
        raise Exception(f"Failed to download {file_type} from {url}: {str(e)}")


def compose_video(
    base_video_path: str,
    overlay_video_path: str,
    audio_path: str,
    output_path: str,
    overlay_opacity: float,
    background_volume: float,
    base_video_volume: float,
    job_id: str
) -> None:
    """
    Compose video with auto-scaling overlay
    
    Process:
    1. Get base video dimensions
    2. Scale overlay to match base video
    3. Apply opacity to overlay
    4. Overlay on base video
    5. Mix audio streams
    """
    
    logger.info(f"[{job_id}] Starting video composition...")
    
    # Get video information
    base_info = get_video_info(base_video_path)
    overlay_info = get_video_info(overlay_video_path)
    
    base_width = base_info['width']
    base_height = base_info['height']
    base_duration = base_info['duration']
    has_audio = base_info['has_audio']
    
    logger.info(f"[{job_id}] Base video: {base_width}×{base_height}, {base_duration:.1f}s, audio={has_audio}")
    logger.info(f"[{job_id}] Overlay video: {overlay_info['width']}×{overlay_info['height']}")
    logger.info(f"[{job_id}] Overlay will be scaled to {base_width}×{base_height}")
    logger.info(f"[{job_id}] Overlay opacity: {overlay_opacity}, BG volume: {background_volume}")
    
    # Update job status
    with jobs_lock:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 15
        jobs[job_id]['video_info'] = {
            'base_resolution': f"{base_width}×{base_height}",
            'overlay_resolution': f"{overlay_info['width']}×{overlay_info['height']}",
            'duration_seconds': base_duration,
            'has_audio': has_audio
        }
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg', '-y',
        '-i', base_video_path,           # Input 0: Base video
        '-stream_loop', '-1', '-i', overlay_video_path,  # Input 1: Overlay (looped)
        '-stream_loop', '-1', '-i', audio_path,          # Input 2: Background audio (looped)
    ]
    
    # Build filter_complex with auto-scaling
    if has_audio:
        # Base video has audio - mix with background
        filter_complex = (
            # Scale overlay to match base video, apply opacity
            f"[1:v]scale={base_width}:{base_height}:flags=lanczos,"
            f"format=yuva420p,colorchannelmixer=aa={overlay_opacity}[overlay];"
            # Overlay on base video
            f"[0:v][overlay]overlay=0:0:shortest=1[vout];"
            # Mix audio streams
            f"[0:a]volume={base_video_volume}[a0];"
            f"[2:a]volume={background_volume}[a1];"
            f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
        )
    else:
        # Base video has NO audio - only background
        filter_complex = (
            f"[1:v]scale={base_width}:{base_height}:flags=lanczos,"
            f"format=yuva420p,colorchannelmixer=aa={overlay_opacity}[overlay];"
            f"[0:v][overlay]overlay=0:0:shortest=1[vout];"
            f"[2:a]volume={background_volume}[aout]"
        )
    
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[vout]',
        '-map', '[aout]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '21',
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'high',
        '-level', '5.1',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        '-threads', '0',
        '-shortest',  # Stop when base video ends
        output_path
    ])
    
    logger.info(f"[{job_id}] FFmpeg command built")
    
    # Execute FFmpeg
    try:
        with jobs_lock:
            jobs[job_id]['progress'] = 20
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Monitor progress
        for line in process.stderr:
            if 'time=' in line and base_duration > 0:
                try:
                    time_str = line.split('time=')[1].split()[0]
                    h, m, s = time_str.split(':')
                    current_sec = int(h) * 3600 + int(m) * 60 + float(s)
                    progress = min(int((current_sec / base_duration) * 75) + 20, 95)
                    
                    with jobs_lock:
                        jobs[job_id]['progress'] = progress
                    
                    if progress % 10 == 0:
                        logger.info(f"[{job_id}] Encoding: {progress}%")
                except:
                    pass
        
        process.wait()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg failed with return code {process.returncode}")
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"[{job_id}] ✅ Composition complete! Size: {file_size_mb:.1f}MB")
        
        with jobs_lock:
            jobs[job_id]['status'] = 'complete'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['output_path'] = output_path
            jobs[job_id]['file_size_mb'] = round(file_size_mb, 2)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{job_id}] ❌ Composition failed: {error_msg}")
        
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = error_msg


def process_job(job_id: str, request: ComposeRequest):
    """
    Background job processor
    
    Steps:
    1. Download all files
    2. Compose video
    3. Cleanup temp files (kept for download)
    """
    temp_dir = None
    
    try:
        logger.info(f"[{job_id}] 🚀 Job started")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
        logger.info(f"[{job_id}] Temp dir: {temp_dir}")
        
        with jobs_lock:
            jobs[job_id]['temp_dir'] = temp_dir
            jobs[job_id]['progress'] = 1
        
        # Define file paths
        base_video_file = os.path.join(temp_dir, "base_video.mp4")
        overlay_file = os.path.join(temp_dir, "overlay.mp4")
        audio_file = os.path.join(temp_dir, "background_audio.mp3")
        output_file = os.path.join(temp_dir, f"composed_{job_id}.mp4")
        
        # Download files
        with jobs_lock:
            jobs[job_id]['message'] = 'Downloading base video...'
        download_file(request.base_video_url, base_video_file, job_id, "base_video")
        
        with jobs_lock:
            jobs[job_id]['progress'] = 5
            jobs[job_id]['message'] = 'Downloading overlay...'
        download_file(request.overlay_video_url, overlay_file, job_id, "overlay")
        
        with jobs_lock:
            jobs[job_id]['progress'] = 10
            jobs[job_id]['message'] = 'Downloading audio...'
        download_file(request.background_audio_url, audio_file, job_id, "audio")
        
        with jobs_lock:
            jobs[job_id]['progress'] = 15
            jobs[job_id]['message'] = 'Composing video...'
        
        # Compose video
        compose_video(
            base_video_file,
            overlay_file,
            audio_file,
            output_file,
            request.overlay_opacity,
            request.background_volume,
            request.base_video_volume,
            job_id
        )
        
        logger.info(f"[{job_id}] ✅ Job completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{job_id}] ❌ Job failed: {error_msg}")
        
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = error_msg
            jobs[job_id]['message'] = f'Failed: {error_msg[:100]}'


def cleanup_old_jobs():
    """Remove jobs older than MAX_JOB_AGE_HOURS"""
    cutoff = datetime.now() - timedelta(hours=MAX_JOB_AGE_HOURS)
    
    with jobs_lock:
        to_delete = []
        for job_id, job in jobs.items():
            try:
                created = datetime.fromisoformat(job['created_at'])
                if created < cutoff:
                    # Delete temp directory
                    temp_dir = job.get('temp_dir')
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        logger.info(f"Cleaned up old job: {job_id}")
                    to_delete.append(job_id)
            except Exception as e:
                logger.error(f"Cleanup error for {job_id}: {e}")
        
        for job_id in to_delete:
            del jobs[job_id]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("=" * 60)
    logger.info("🚀 Starting Video Composer API v2.0.0")
    logger.info("=" * 60)
    
    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.decode().split('\n')[0]
            logger.info(f"✅ FFmpeg available: {version}")
        else:
            logger.error("❌ FFmpeg not found!")
    except Exception as e:
        logger.error(f"❌ FFmpeg check failed: {e}")
    
    # Start cleanup worker
    def cleanup_worker():
        while True:
            time.sleep(3600)  # Every hour
            cleanup_old_jobs()
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("✅ Cleanup worker started")
    logger.info("=" * 60)


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Video Composer API",
        "version": "2.0.0",
        "description": "Async video composition with auto-scaling overlay support",
        "features": [
            "Auto-scales overlay to match base video resolution",
            "Supports videos up to 4K (3840×2160)",
            "Handles 2+ hour videos",
            "Async processing with progress tracking",
            "Configurable overlay opacity (0-100%)",
            "Configurable audio volumes"
        ],
        "endpoints": {
            "POST /compose": "Start new composition job",
            "GET /status/{job_id}": "Check job status and progress",
            "GET /download/{job_id}": "Download completed video",
            "DELETE /job/{job_id}": "Delete job and cleanup files",
            "GET /health": "Health check"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    active_jobs = [j for j in jobs.values() if j['status'] in ['pending', 'processing']]
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(active_jobs),
        "total_jobs": len(jobs)
    }


@app.post("/compose", response_model=JobResponse)
async def compose(request: ComposeRequest, background_tasks: BackgroundTasks):
    """
    Start a new video composition job
    
    Returns job_id for status tracking
    """
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create job entry
    with jobs_lock:
        jobs[job_id] = {
            'job_id': job_id,
            'status': 'pending',
            'progress': 0,
            'message': 'Job created',
            'created_at': datetime.now().isoformat(),
            'request': request.dict(),
            'error': None,
            'completed_at': None,
            'output_path': None,
            'temp_dir': None,
            'file_size_mb': None,
            'video_info': None
        }
    
    # Start background processing
    background_tasks.add_task(process_job, job_id, request)
    
    logger.info(f"[{job_id}] Job created and queued")
    
    return JobResponse(
        job_id=job_id,
        status='pending',
        message='Job created successfully. Use status endpoint to track progress.',
        status_url=f"/status/{job_id}"
    )


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """
    Get job status and progress
    
    Poll this endpoint to track job progress (0-100%)
    """
    
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
    
    response = StatusResponse(
        job_id=job_id,
        status=job['status'],
        progress=job.get('progress', 0),
        message=job.get('message'),
        error=job.get('error'),
        created_at=job['created_at'],
        completed_at=job.get('completed_at'),
        file_size_mb=job.get('file_size_mb'),
        video_info=job.get('video_info')
    )
    
    if job['status'] == 'complete':
        response.download_url = f"/download/{job_id}"
    
    return response


@app.get("/download/{job_id}")
async def download(job_id: str):
    """
    Download completed video
    
    Returns MP4 file
    """
    
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
    
    if job['status'] != 'complete':
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {job['status']}"
        )
    
    output_path = job.get('output_path')
    
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"composed_{job_id}.mp4"
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete job and cleanup files
    
    Use this to free up disk space
    """
    
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        
        # Delete temp directory
        temp_dir = job.get('temp_dir')
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"[{job_id}] Deleted temp directory")
            except Exception as e:
                logger.error(f"[{job_id}] Cleanup error: {e}")
        
        del jobs[job_id]
    
    logger.info(f"[{job_id}] Job deleted")
    
    return {"message": "Job deleted successfully", "job_id": job_id}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global error handler"""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info"
    )
