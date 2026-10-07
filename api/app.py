import asyncio
import os
import time
import json
from typing import Optional, List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pipeline import VisionPipeline

app = FastAPI(
    title="Intelligent Railway Crowd Monitoring & Anomaly Detection System",
    description="Mission-Control Surveillance Telemetry & Real-Time Computer Vision Pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Vision Pipeline instance
pipeline: Optional[VisionPipeline] = None

# Active WebSocket connections
active_websockets: List[WebSocket] = []

class SourceSwitchRequest(BaseModel):
    source: str

@app.on_event("startup")
async def startup_event():
    global pipeline
    loop = asyncio.get_running_loop()
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    pipeline = VisionPipeline(config_path)
    pipeline.start()
    
    # Broadcast callback for pipeline telemetry
    def on_telemetry(data):
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_broadcast_telemetry(data), loop)
        
    pipeline.telemetry_listeners.append(on_telemetry)

@app.on_event("shutdown")
async def shutdown_event():
    global pipeline
    if pipeline:
        pipeline.stop()

async def _broadcast_telemetry(data: dict):
    for ws in list(active_websockets):
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            if ws in active_websockets:
                active_websockets.remove(ws)

# Mount dashboard static assets
app.mount("/static", StaticFiles(directory="dashboard"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard_root():
    index_file = "dashboard/index.html"
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>Railway Crowd Monitoring Dashboard loading...</h1>")

@app.get("/api/status")
async def get_status():
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {
        "status": "online",
        "station": pipeline.config.camera.station_name,
        "camera_id": pipeline.config.camera.camera_id,
        "source": str(pipeline.current_source),
        "fps": pipeline.current_fps,
        "frame_idx": pipeline.current_frame_idx,
        "device": pipeline.detector.device,
        "anonymization_enabled": pipeline.config.privacy.anonymize_faces
    }

@app.get("/api/sources")
async def get_available_sources():
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {
        "current": str(pipeline.current_source),
        "options": [
            {"id": "0", "name": "📹 Live Laptop / USB Webcam (Camera 0)"},
            {"id": "data/station_entry_live.mp4", "name": "🚂 Station Entry Live Stream (Live #train)"},
            {"id": "data/crowd_in_railway_station.mp4", "name": "👥 Station Concourse CCTV (Crowd)"},
            {"id": "data/synthetic_sample_cctv.mp4", "name": "⚙️ Synthetic Simulation Benchmark"}
        ]
    }

@app.post("/api/source")
async def change_camera_source(req: SourceSwitchRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    ok = pipeline.change_source(req.source)
    return {"status": "ok", "current_source": str(pipeline.current_source)}

@app.get("/api/metrics")
async def get_metrics():
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline.get_latest_telemetry()

@app.get("/api/alerts")
async def get_alerts(limit: int = 50):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline.alert_manager.get_recent_alerts(limit=limit)

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    ok = pipeline.alert_manager.acknowledge_alert(alert_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Alert ID not found")
    return {"status": "acknowledged", "id": alert_id}

@app.post("/api/toggle/{setting}")
async def toggle_setting(setting: str):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if setting == "heatmap":
        pipeline.show_heatmap = not pipeline.show_heatmap
        return {"heatmap": pipeline.show_heatmap}
    elif setting == "polygons":
        pipeline.show_polygons = not pipeline.show_polygons
        return {"polygons": pipeline.show_polygons}
    elif setting == "tracks":
        pipeline.show_tracks = not pipeline.show_tracks
        return {"tracks": pipeline.show_tracks}
    elif setting == "privacy":
        pipeline.show_anonymization = not pipeline.show_anonymization
        return {"privacy": pipeline.show_anonymization}
    raise HTTPException(status_code=400, detail="Invalid toggle setting")

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    if pipeline:
        try:
            await websocket.send_text(json.dumps(pipeline.get_latest_telemetry()))
        except Exception:
            pass
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

def _generate_mjpeg_stream():
    while pipeline and pipeline.is_running:
        frame_bytes = pipeline.get_latest_frame()
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04) # ~25 fps

@app.get("/api/stream")
async def video_stream():
    return StreamingResponse(
        _generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)
