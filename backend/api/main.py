from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.api.data_loader import load_data
from backend.api.routes import spills, vessels, ais, ocean, pipeline_ws, demo, incidents

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading data...")
    load_data()
    print("Data loaded successfully.")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(spills.router, prefix="/api/spills", tags=["spills"])
app.include_router(vessels.router, prefix="/api/vessels", tags=["vessels"])
app.include_router(ais.router, prefix="/api/ais", tags=["ais"])
app.include_router(ocean.router, prefix="/api/ocean", tags=["ocean"])
app.include_router(demo.router, prefix="/api", tags=["demo"])
app.include_router(incidents.router, prefix="/api", tags=["incidents"])
app.include_router(pipeline_ws.router, prefix="/ws/pipeline", tags=["pipeline"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
