"""
FastAPI Async Video Composer - 4K Support with Overlay Opacity
POST /compose → krijg job_id
GET /status/{job_id} → check status
GET /download/{job_id} → download video
"""

import os
import subprocess
import tempfile
import logging
import uuid
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import shutil

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Video Composer 4K", version="1.0.0")

# In-memory job storage
jobs: Dict[str, dict] = {}
jobs_lock = threading.Lock()

# Cleanup oude jobs na 48 uur
MAX_JOB_AGE_HOURS = 48


# ============================================================================
# MODELS
# ============================================================================

class ComposeRequest(BaseModel):
    video_url: str = Field(..., description="Base video URL (MP4)")
    overlay_url: str = Field(..., description="Overlay video URL (MP4)")
    audio_url: str = Field(..., description="Ambience audio URL (MP3)")
    video_volume: float = Field(default=1.0, ge=0.0, le=2.0, description="Video audio volume")
    ambience_volume: float = Field(default=0.15, ge=0.0, le=2.0, description="Ambience volume")
    overlay_opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Overlay opacity: 1.0=visible, 0.0=invisible")
    return_mode: str = Field(default="async", description="Always async for long videos")


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    download_url: Optional[str] = None
    file_size_mb: Optional[float] = None


# ============================================================================
# HELPERS
# ============================================================================

def cleanup_old_jobs():
    """Verwijder jobs ouder dan MAX_JOB_AGE_HOURS"""
    cutoff = datetime.now() - timedelta(hours=MAX_JOB_AGE_HOURS)
    
    with jobs_lock:
        to_delete = []
        for job_id, job in jobs.items():
            created = datetime.fromisoformat(job['created_at'])
            if created < cutoff:
                # Verwijder output file
                if job.get('output_path') and os.path.exists(job['output_path']):
                    try:
                        os.remove(job['output_path'])
                        logger.info(f"Deleted old file: {job['output_path']}")
                    except Exception as e:
                        logger.error(f"Failed to delete {job['output_path']}: {e}")
                
                # Verwijder temp dir
                if job.get('temp_dir') and os.path.exists(job['temp_dir']):
                    try:
                        shutil.rmtree(job['temp_dir'])
                        logger.info(f"Deleted temp dir: {job['temp_dir']}")
                    except Exception as e:
                        logger.error(f"Failed to delete temp dir: {e}")
                
                to_delete.append(job_id)
        
        for job_id in to_delete:
            del jobs[job_id]
            logger.info(f"Cleaned up old job: {job_id}")


def download_file(url: str, dest_path: str, timeout: int = 900) -> str:
    """Download file met progress logging"""
    logger.info(f"Downloading: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):  # 2MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        progress = int((downloaded / total_size) * 100)
                        if progress % 20 == 0:
                            logger.info(f"Download progress: {progress}%")
        
        file_size = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"Downloaded: {file_size:.1f}MB")
        return dest_path
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise Exception(f"Download mislukt voor {url}: {str(e)}")


def check_audio_stream(video_path: str) -> bool:
    """Check of video audio heeft"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() == 'audio'
    except Exception:
        return False


def get_video_info(video_path: str) -> Dict:
    """Haal video metadata op"""
    try:
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
        
        import json
        data = json.loads(result.stdout)
        stream = data.get('streams', [{}])[0]
        
        return {
            'width': stream.get('width', 1920),
            'height': stream.get('height', 1080),
            'duration': float(stream.get('duration', 0))
        }
    except Exception as e:
        logger.warning(f"Kon video info niet ophalen: {e}")
        return {'width': 1920, 'height': 1080, 'duration': 0}


def compose_video_4k(
    video_path: str,
    overlay_path: str,
    audio_path: str,
    output_path: str,
    video_volume: float,
    ambience_volume: float,
    overlay_opacity: float,
    job_id: str
) -> None:
    """4K video compositie met overlay opacity en audio mix"""
    
    logger.info(f"[{job_id}] Starting 4K video composition...")
    
    # Haal video info op
    video_info = get_video_info(video_path)
    has_audio = check_audio_stream(video_path)
    
    logger.info(f"[{job_id}] Video: {video_info['width']}x{video_info['height']}, duration: {video_info['duration']:.1f}s")
    logger.info(f"[{job_id}] Audio: {has_audio}, Overlay opacity: {overlay_opacity}")
    
    # Update job status
    with jobs_lock:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 10
        jobs[job_id]['video_info'] = video_info
    
    # Bouw FFmpeg command
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-stream_loop', '-1', '-i', overlay_path,
        '-stream_loop', '-1', '-i', audio_path,
    ]
    
    # Filtergraph met opacity control
    if has_audio:
        filter_complex = (
            f"[1:v]format=yuva420p,colorchannelmixer=aa={overlay_opacity}[overlay];"
            f"[0:v][overlay]overlay=0:0:shortest=1[vout];"
            f"[0:a]volume={video_volume}[a0];"
            f"[2:a]volume={ambience_volume}[a1];"
            f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
        )
    else:
        filter_complex = (
            f"[1:v]format=yuva420p,colorchannelmixer=aa={overlay_opacity}[overlay];"
            f"[0:v][overlay]overlay=0:0:shortest=1[vout];"
            f"[2:a]volume={ambience_volume}[aout]"
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
        '-shortest',
        output_path
    ])
    
    logger.info(f"[{job_id}] FFmpeg command ready")
    
    # Voer FFmpeg uit
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
        duration = video_info['duration']
        
        for line in process.stderr:
            if 'time=' in line and duration > 0:
                try:
                    time_str = line.split('time=')[1].split()[0]
                    h, m, s = time_str.split(':')
                    current_sec = int(h) * 3600 + int(m) * 60 + float(s)
                    progress = min(int((current_sec / duration) * 80) + 20, 99)
                    
                    with jobs_lock:
                        jobs[job_id]['progress'] = progress
                    
                    if progress % 10 == 0:
                        logger.info(f"[{job_id}] Encoding progress: {progress}%")
                except:
                    pass
        
        process.wait()
        
        if process.returncode != 0:
            stderr_output = process.stderr.read() if hasattr(process.stderr, 'read') else "Unknown error"
            raise Exception(f"FFmpeg failed: {stderr_output[-500:]}")
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"[{job_id}] Complete! Size: {file_size_mb:.1f}MB")
        
        with jobs_lock:
            jobs[job_id]['status'] = 'complete'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['output_path'] = output_path
            jobs[job_id]['file_size_mb'] = file_size_mb
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{job_id}] Failed: {error_msg}")
        
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = error_msg


def process_job(job_id: str, request: ComposeRequest):
    """Background job processor"""
    temp_dir = None
    
    try:
        logger.info(f"[{job_id}] Starting job...")
        
        temp_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
        logger.info(f"[{job_id}] Temp dir: {temp_dir}")
        
        with jobs_lock:
            jobs[job_id]['temp_dir'] = temp_dir
        
        # Download files
        video_file = os.path.join(temp_dir, "video.mp4")
        overlay_file = os.path.join(temp_dir, "overlay.mp4")
        audio_file = os.path.join(temp_dir, "audio.mp3")
        output_file = os.path.join(temp_dir, f"output_{job_id}.mp4")
        
        logger.info(f"[{job_id}] Downloading video...")
        download_file(request.video_url, video_file)
        
        with jobs_lock:
            jobs[job_id]['progress'] = 5
        
        logger.info(f"[{job_id}] Downloading overlay...")
        download_file(request.overlay_url, overlay_file)
        
        with jobs_lock:
            jobs[job_id]['progress'] = 8
        
        logger.info(f"[{job_id}] Downloading audio...")
        download_file(request.audio_url, audio_file)
        
        with jobs_lock:
            jobs[job_id]['progress'] = 10
        
        # Compose video
        compose_video_4k(
            video_file,
            overlay_file,
            audio_file,
            output_file,
            request.video_volume,
            request.ambience_volume,
            request.overlay_opacity,
            job_id
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{job_id}] Job failed: {error_msg}")
        
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = error_msg


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("Starting Video Composer API (4K Support)...")
    
    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            logger.info("✓ FFmpeg available")
        else:
            logger.error("✗ FFmpeg not found!")
    except Exception as e:
        logger.error(f"✗ FFmpeg check failed: {e}")
    
    # Start cleanup thread
    def cleanup_worker():
        while True:
            time.sleep(3600)
            cleanup_old_jobs()
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("✓ Cleanup worker started")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Video Composer API - 4K Support",
        "version": "1.0.0",
        "features": [
            "4K video support (up to 3840x2160)",
            "2+ hour videos supported",
            "Async job processing",
            "Adjustable overlay opacity",
            "Progress tracking"
        ],
        "endpoints": {
            "health": "GET /health",
            "compose": "POST /compose",
            "status": "GET /status/{job_id}",
            "download": "GET /download/{job_id}"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "ok": True,
        "active_jobs": len([j for j in jobs.values() if j['status'] in ['pending', 'processing']]),
        "total_jobs": len(jobs)
    }


@app.post("/compose")
async def compose(request: ComposeRequest, background_tasks: BackgroundTasks):
    """Start video composition job"""
    
    job_id = str(uuid.uuid4())
    
    with jobs_lock:
        jobs[job_id] = {
            'job_id': job_id,
            'status': 'pending',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'request': request.dict(),
            'error': None,
            'completed_at': None,
            'output_path': None,
            'temp_dir': None,
            'file_size_mb': None,
            'video_info': None
        }
    
    background_tasks.add_task(process_job, job_id, request)
    
    logger.info(f"[{job_id}] Job created")
    
    return JobResponse(
        job_id=job_id,
        status='pending',
        message='Job created. Use GET /status/{job_id} to check progress.'
    )


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Get job status"""
    
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
    
    response = StatusResponse(
        job_id=job_id,
        status=job['status'],
        progress=job.get('progress', 0),
        error=job.get('error'),
        created_at=job['created_at'],
        completed_at=job.get('completed_at'),
        file_size_mb=job.get('file_size_mb')
    )
    
    if job['status'] == 'complete':
        response.download_url = f"/download/{job_id}"
    
    return response


@app.get("/download/{job_id}")
async def download(job_id: str):
    """Download completed video"""
    
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
    
    if job['status'] != 'complete':
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Status: {job['status']}"
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
    """Delete job and cleanup files"""
    
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        
        temp_dir = job.get('temp_dir')
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"[{job_id}] Deleted temp dir")
            except Exception as e:
                logger.error(f"[{job_id}] Cleanup error: {e}")
        
        del jobs[job_id]
    
    logger.info(f"[{job_id}] Job deleted")
    
    return {"message": "Job deleted", "job_id": job_id}


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
