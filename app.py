"""
app.py
Single entrypoint that wires everything together.

Run with: uvicorn app:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.orchestrator import Orchestrator
from core.config import settings
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    orchestrator = Orchestrator()
    await orchestrator.start()
    app.state.orchestrator = orchestrator
    yield
    await orchestrator.stop()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# The Angular frontend runs on its own dev server (a different origin/port
# than this API), so the browser blocks requests to it by default unless we
# explicitly allow it here (CORS). Without this, the frontend gets a vague
# "0 Unknown Error" instead of a real response.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to the exact frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)