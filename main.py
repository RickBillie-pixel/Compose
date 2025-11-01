"""
Video Composer API - Direct Video URL Support
==============================================
POST /compose → Start job, krijg job_id
GET /status/{job_id} → Check status, krijg video_url
GET /videos/{filename} → Direct video access (voor fal.ai)

Features:
- Direct video URL zoals foto-verlenger API
- Auto-scales overlay to base video
- Async processing met progress tracking
- 48 uur video beschikbaarheid
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
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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
    version="3.0.0",
    description="Video composition with direct URL output - Perfect for fal.ai!"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ============================================================================
# DIRECTORIES - Gebruik /data voor Render persistent storage
# ============================================================================

if os.path.exists('/data'):
    BASE_DIR = Path('/data')
    logger.info("🗄️  Using Render persistent disk: /data")
else:
    BASE_DIR = Path(tempfile.gettempdir())
    logger.info("⚠️  Using temporary storage (local dev)")

TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "videos"
JOBS_DIR = BASE_DIR / "jobs"

TEMP_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
JOBS_DIR.mkdir(exist_ok=True, parents=True)

# Mount static files voor directe video toegang
app.mount("/videos", StaticFiles(directory=str(OUTPUT_DIR)), name="videos")

logger.info("=" * 80)
logger.info("🚀 VIDEO COMPOSER API - DIRECT URL OUTPUT")
logger.info("=" * 80)
logger.info(f"📁 Temp directory: {TEMP_DIR}")
logger.info(f"📁 Output directory: {OUTPUT_DIR}")
logger.info(f"📁 Jobs directory: {JOBS_DIR}")
logger.info(f"🌐 Video serving: /videos (static files)")
logger.info("=" * 80)

# ============================================================================
# CLEANUP THREAD
# ============================================================================

def cleanup_old_files():
    """Verwijdert videos ouder dan 48 uur"""
    while True:
        try:
            time.sleep(3600)  # Check elk uur
            cutoff = datetime.now() - timedelta(hours=48)
            
            deleted_count = 0
            for video_file in OUTPUT_DIR.glob("*.mp4"):
                file_time = datetime.fromtimestamp(video_file.stat().st_mtime)
                if file_time < cutoff:
                    video_file.unlink()
                    deleted_count += 1
                    
                    # Verwijder job info
                    job_id = video_file.stem
                    job_file = JOBS_DIR / f"{job_id}.json"
                    if job_file.exists():
                        job_file.unlink()
            
            # Cleanup temp files
            for temp_dir in TEMP_DIR.glob("job_*"):
                dir_time = datetime.fromtimestamp(temp_dir.stat().st_mtime)
                if dir_time < cutoff:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} old files")
                
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

# ============================================================================
# JOB MANAGEMENT
# ============================================================================

def update_job_status(job_id: str, **kwargs):
    """Update job status"""
    job_file = JOBS_DIR / f"{job_id}.json"
    
    if job_file.exists():
        with open(job_file, 'r') as f:
            job_info = json.load(f)
    else:
        job_info = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat()
        }
    
    job_info["updated_at"] = datetime.now().isoformat()
    job_info.update(kwargs)
    
    with open(job_file, 'w') as f:
        json.dump(job_info, f, indent=2)


def get_job_info(job_id: str) -> Optional[Dict]:
    """Haal job info op"""
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        return None
    
    try:
        with open(job_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Kan job niet lezen: {e}")
        return None

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ComposeRequest(BaseModel):
    """Request model"""
    base_video_url: str = Field(..., description="Base video URL (MP4)")
    overlay_video_url: str = Field(..., description="Overlay video URL (MP4)")
    background_audio_url: str = Field(..., description="Background audio URL (MP3)")
    overlay_opacity: float = Field(default=0.25, ge=0.0, le=1.0)
    background_volume: float = Field(default=0.15, ge=0.0, le=1.0)
    base_video_volume: float = Field(default=1.0, ge=0.0, le=2.0)

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def get_video_info(video_path: str) -> Dict:
    """Get video metadata"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        data = json.loads(result.stdout)
        stream = data.get('streams', [{}])[0]
        
        width = stream.get('width', 1920)
        height = stream.get('height', 1080)
        duration = float(stream.get('duration', 0))
        
        # Check audio
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
        return {'width': 1920, 'height': 1080, 'duration': 0, 'has_audio': False}


def download_file(url: str, dest_path: str, job_id: str, file_type: str) -> None:
    """Download file with progress"""
    logger.info(f"[{job_id[:8]}] Downloading {file_type}...")
    
    try:
        response = requests.get(url, stream=True, timeout=900)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        file_size = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"[{job_id[:8]}] Downloaded {file_type}: {file_size:.1f}MB")
        
    except Exception as e:
        raise Exception(f"Download failed for {file_type}: {str(e)}")


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
    """Compose video with auto-scaling"""
    
    logger.info(f"[{job_id[:8]}] Starting composition...")
    
    base_info = get_video_info(base_video_path)
    overlay_info = get_video_info(overlay_video_path)
    
    base_width = base_info['width']
    base_height = base_info['height']
    base_duration = base_info['duration']
    has_audio = base_info['has_audio']
    
    logger.info(f"[{job_id[:8]}] Base: {base_width}×{base_height}, {base_duration:.1f}s")
    logger.info(f"[{job_id[:8]}] Overlay: {overlay_info['width']}×{overlay_info['height']} → scaling to {base_width}×{base_height}")
    
    update_job_status(
        job_id,
        status='processing',
        progress=15,
        video_info={
            'base_resolution': f"{base_width}×{base_height}",
            'overlay_resolution': f"{overlay_info['width']}×{overlay_info['height']}",
            'duration_seconds': base_duration
        }
    )
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg', '-y',
        '-i', base_video_path,
        '-stream_loop', '-1', '-i', overlay_video_path,
        '-stream_loop', '-1', '-i', audio_path,
    ]
    
    if has_audio:
        filter_complex = (
            f"[1:v]scale={base_width}:{base_height}:flags=lanczos,"
            f"format=yuva420p,colorchannelmixer=aa={overlay_opacity}[overlay];"
            f"[0:v][overlay]overlay=0:0:shortest=1[vout];"
            f"[0:a]volume={base_video_volume}[a0];"
            f"[2:a]volume={background_volume}[a1];"
            f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
        )
    else:
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
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        '-shortest',
        output_path
    ])
    
    # Execute FFmpeg
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
                progress = min(int((current_sec / base_duration) * 80) + 15, 95)
                
                update_job_status(job_id, progress=progress)
                
                if progress % 10 == 0:
                    logger.info(f"[{job_id[:8]}] Progress: {progress}%")
            except:
                pass
    
    process.wait()
    
    if process.returncode != 0:
        raise Exception("FFmpeg failed")
    
    if not os.path.exists(output_path):
        raise Exception("Output file not created")


def process_job(job_id: str, request: ComposeRequest, base_url: str):
    """Background job processor"""
    
    # Create job temp dir
    job_temp_dir = TEMP_DIR / f"job_{job_id}"
    job_temp_dir.mkdir(exist_ok=True)
    
    try:
        logger.info(f"[{job_id[:8]}] 🚀 Job started")
        
        update_job_status(
            job_id,
            status='processing',
            progress=0,
            message='Downloading files...'
        )
        
        # File paths
        base_video_file = str(job_temp_dir / "base_video.mp4")
        overlay_file = str(job_temp_dir / "overlay.mp4")
        audio_file = str(job_temp_dir / "audio.mp3")
        output_file = str(OUTPUT_DIR / f"{job_id}.mp4")  # Gebruik job_id als filename!
        
        # Download files
        download_file(request.base_video_url, base_video_file, job_id, "base_video")
        update_job_status(job_id, progress=5)
        
        download_file(request.overlay_video_url, overlay_file, job_id, "overlay")
        update_job_status(job_id, progress=10)
        
        download_file(request.background_audio_url, audio_file, job_id, "audio")
        update_job_status(job_id, progress=15)
        
        # Compose video
        start_time = datetime.now()
        
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
        
        processing_time = (datetime.now() - start_time).total_seconds()
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        expires_at = datetime.now() + timedelta(hours=48)
        
        # BELANGRIJKSTE: Genereer directe video URL
        video_url = f"{base_url}/videos/{job_id}.mp4"
        
        logger.info("=" * 80)
        logger.info(f"✅ JOB COMPLETED [{job_id[:8]}]")
        logger.info(f"   ⏱️  Time: {processing_time:.1f}s")
        logger.info(f"   📦 Size: {file_size_mb:.1f}MB")
        logger.info(f"   🔗 Video URL: {video_url}")
        logger.info(f"   ⏰ Expires: {expires_at.isoformat()}")
        logger.info("=" * 80)
        
        update_job_status(
            job_id,
            status='completed',
            progress=100,
            message='Video ready!',
            processing_time_seconds=round(processing_time, 1),
            file_size_mb=round(file_size_mb, 2),
            video_url=video_url,  # DIRECTE VIDEO URL!
            expires_at=expires_at.isoformat(),
            expires_in_hours=48,
            completed_at=datetime.now().isoformat()
        )
        
        # Cleanup temp dir
        shutil.rmtree(job_temp_dir, ignore_errors=True)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{job_id[:8]}] ❌ Failed: {error_msg}")
        
        update_job_status(
            job_id,
            status='failed',
            error=error_msg,
            message=f'Failed: {error_msg[:100]}'
        )
        
        # Cleanup
        shutil.rmtree(job_temp_dir, ignore_errors=True)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Starting Video Composer API...")
    
    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            logger.info("✅ FFmpeg available")
    except Exception as e:
        logger.error(f"❌ FFmpeg check failed: {e}")


@app.get("/")
async def root():
    """API info"""
    return {
        "name": "Video Composer API",
        "version": "3.0.0 - Direct URL Output",
        "status": "🟢 Online",
        "workflow": {
            "step_1": "POST /compose → Krijg job_id",
            "step_2": "GET /status/{job_id} → Poll status (krijg video_url when ready)",
            "step_3": "Gebruik video_url direct in fal.ai!"
        },
        "endpoints": {
            "POST /compose": "Start composition job",
            "GET /status/{job_id}": "Check status + get video_url",
            "GET /videos/{filename}": "Direct video access",
            "GET /health": "Health check"
        },
        "features": [
            "✅ Direct video URL - Perfect voor fal.ai!",
            "✅ Auto-scales overlay to base video",
            "✅ 48 uur video beschikbaarheid",
            "✅ Supports 4K videos",
            "✅ 2+ hour videos supported"
        ]
    }


@app.get("/health")
async def health():
    """Health check"""
    total_jobs = len(list(JOBS_DIR.glob("*.json")))
    total_videos = len(list(OUTPUT_DIR.glob("*.mp4")))
    
    return {
        "status": "healthy",
        "ffmpeg": "available",
        "total_jobs": total_jobs,
        "total_videos": total_videos,
        "storage": "persistent" if os.path.exists('/data') else "ephemeral"
    }


@app.post("/compose")
async def compose(request: ComposeRequest, background_tasks: BackgroundTasks, http_request: Request):
    """
    Start video composition job
    
    Returns job_id → Poll /status/{job_id} → Get video_url
    """
    
    job_id = str(uuid.uuid4())
    
    # Get base URL
    base_url = str(http_request.base_url).rstrip('/')
    
    logger.info("=" * 80)
    logger.info(f"📥 NEW JOB [ID: {job_id[:8]}]")
    logger.info(f"   Base URL: {base_url}")
    logger.info("=" * 80)
    
    # Create job
    update_job_status(
        job_id,
        status='pending',
        progress=0,
        request_params={
            'base_video_url': request.base_video_url,
            'overlay_video_url': request.overlay_video_url,
            'background_audio_url': request.background_audio_url,
            'overlay_opacity': request.overlay_opacity,
            'background_volume': request.background_volume
        }
    )
    
    # Start background task
    background_tasks.add_task(process_job, job_id, request, base_url)
    
    logger.info(f"✅ Job queued: {job_id[:8]}")
    logger.info(f"🔗 Status URL: {base_url}/status/{job_id}")
    logger.info(f"🔗 Video URL: {base_url}/videos/{job_id}.mp4 (after completion)")
    
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "message": "Video composition started - Poll /status/{job_id} for video_url",
            "job_id": job_id,
            "status_url": f"/status/{job_id}",
            "poll_interval_seconds": 10
        }
    )


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Get job status
    
    Wanneer completed: krijg video_url!
    """
    
    job_info = get_job_info(job_id)
    
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    logger.info(f"📊 Status check: {job_id[:8]} - {job_info.get('status')} - {job_info.get('progress', 0)}%")
    
    return job_info


@app.get("/download/{job_id}")
async def download(job_id: str):
    """
    Download video (alternatief voor direct URL)
    """
    
    job_info = get_job_info(job_id)
    
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_info.get('status') != 'completed':
        raise HTTPException(status_code=400, detail=f"Job not complete. Status: {job_info.get('status')}")
    
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"composed_{job_id}.mp4"
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
