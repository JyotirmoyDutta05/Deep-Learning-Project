"""FastAPI service + web UI.   uvicorn phishguard.api:app --port 8000"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .extract import extract_text_from_file
from .service import Predictor

STATIC = Path(__file__).parent / "static"
Text = Annotated[str, Field(min_length=1, max_length=20000)]


class AnalyzeRequest(BaseModel):
    text: Text
    channel: Literal["email", "sms", "transaction"] = "sms"
    explain: bool = True


class BatchRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=100)


def create_app(model_path: str | None = None) -> FastAPI:
    path = model_path or os.environ.get("PHISHGUARD_MODEL", "artifacts/lstm/model.pt")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = Predictor(path) if Path(path).exists() else None
        yield

    app = FastAPI(title="PhishSense AI", version=__version__, lifespan=lifespan)

    def predictor() -> Predictor:
        p = app.state.predictor
        if p is None:
            raise HTTPException(503, "No model loaded. Train one: python -m phishguard.train")
        return p

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "model_loaded": app.state.predictor is not None, "version": __version__}

    @app.get("/v1/model")
    def model_info():
        return predictor().info()

    @app.post("/v1/analyze")
    def analyze(req: AnalyzeRequest):
        try:  # message content is never logged
            return {**predictor().analyze(req.text, req.explain), "channel": req.channel}
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.post("/v1/analyze/batch")
    def analyze_batch(req: BatchRequest):
        try:
            return {"results": predictor().score_many(req.texts)}
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.post("/v1/analyze/file")
    async def analyze_file(file: UploadFile = File(...), explain: bool = True):
        content = await file.read()
        extracted, desc, meta = extract_text_from_file(content, file.filename or "unknown.txt")
        if not extracted or extracted.startswith("[Error"):
            raise HTTPException(400, f"Text extraction failed: {extracted}")
        try:
            res = predictor().analyze(extracted, explain)
            res["source_file"] = {"filename": file.filename, "type": desc, **meta}
            return res
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
