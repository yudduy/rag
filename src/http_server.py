import os
import io
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
import httpx

from src.metrics import prometheus_metrics
from src.resource_manager import get_resource_manager
from src.security import validate_file_upload
from src.generate import generate_index


app = FastAPI(title="RAG Auxiliary API", version="0.1.0")


@app.get("/metrics")
def get_metrics() -> Response:
    payload, content_type = prometheus_metrics()
    return Response(content=payload, media_type=content_type)


@app.get("/health")
def health() -> JSONResponse:
    status = get_resource_manager().get_health_status()
    return JSONResponse(status)


def _save_upload(dest_dir: Path, up: UploadFile) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / up.filename
    with out_path.open("wb") as f:
        shutil.copyfileobj(up.file, f)
    return out_path


async def _download_to(dest_dir: Path, url: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = url.split("/")[-1] or "downloaded"
    out_path = dest_dir / name
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content = resp.content
    with out_path.open("wb") as f:
        f.write(content)
    return out_path


@app.post("/ingest")
async def ingest(
    files: Optional[List[UploadFile]] = File(default=None),
    url: Optional[str] = Form(default=None),
):
    data_dir = Path(os.environ.get("DATA_DIR", "web/data"))
    saved: List[str] = []

    if not files and not url:
        raise HTTPException(status_code=400, detail="Provide files or url")

    if files:
        for up in files:
            content = await up.read()
            ok, meta = validate_file_upload(up.filename, content)
            if not ok:
                raise HTTPException(status_code=400, detail={"file": up.filename, "violations": meta.get("violations", [])})
            up.file = io.BytesIO(content)
            path = _save_upload(data_dir, up)
            saved.append(str(path))

    if url:
        if not url.startswith("http://") and not url.startswith("https://"):
            raise HTTPException(status_code=400, detail="url must start with http(s)://")
        try:
            path = await _download_to(data_dir, url)
            # Validate downloaded file by path only (size/ext)
            ok, meta = validate_file_upload(path)
            if not ok:
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail={"url": url, "violations": meta.get("violations", [])})
            saved.append(str(path))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=400, detail=f"download failed: {e}")

    try:
        generate_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"indexing failed: {e}")

    return {"ingested": saved, "index_refreshed": True}


