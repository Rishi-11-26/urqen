"""
backend/main.py
---------------
FastAPI server entrypoint for URQEN Tabular ML Platform.
Exposes REST API, OpenAPI docs, and manages background experimentation tasks.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.routers import datasets, experiments, artifacts, admin, presence, account

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "URQEN Tabular Machine Learning & Decision-Support REST API. "
        "Provides dataset profiling, quality findings, model benchmarking, "
        "experiment results, and downloadable artifacts."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(artifacts.router)
app.include_router(admin.router)
app.include_router(presence.router)
app.include_router(account.router)



@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health and readiness probe."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "storage": {
            "datasets_dir": str(settings.DATA_DIR),
            "artifacts_dir": str(settings.ARTIFACTS_DIR),
        }
    }


import pathlib
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dist = pathlib.Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api") or full_path in ("docs", "redoc", "openapi.json"):
            raise HTTPException(status_code=404, detail="Not found")
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
