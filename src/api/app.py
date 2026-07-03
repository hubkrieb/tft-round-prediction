"""FastAPI service: serve the board-builder UI and score matchups.

Routes
------
* ``GET  /``             - the single-page board builder (static ``index.html``).
* ``GET  /api/models``   - which model backends have weights available on disk.
* ``POST /api/predict``  - score a :class:`BoardState` with the chosen model.
* ``/assets/*`` ``/catalog.json`` - static icons + unit/item/trait catalog.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.api import config
from src.api.predictor import available_models, get_predictor, loaded_models
from src.api.schema import PredictRequest, PredictResponse

DEVICE = os.environ.get("TRP_DEVICE", "cpu")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(title="TFT Round Prediction", version="0.1.0")

    if not config.CATALOG_PATH.exists():
        # Fail loud with an actionable message rather than serving a broken UI.
        print(
            "WARNING: catalog.json is missing. Run `trp fetch-assets` to download "
            "the set 16 icons and build the catalog before using the UI."
        )

    @app.get("/api/models")
    def models() -> dict:
        """Report available/loaded model backends and the active compute device."""
        return {
            "available": available_models(),
            "loaded": loaded_models(),
            "device": DEVICE,
        }

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        """Score a matchup and return the player's win probability."""
        try:
            predictor = get_predictor(req.model, device=DEVICE)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        prob = predictor.predict_proba(req)
        return PredictResponse(
            model=req.model,
            win_probability=prob,
            prediction="victory" if prob >= 0.5 else "defeat",
        )

    # Serve the src/web frontend (index.html at "/", plus /assets and
    # /catalog.json). Mounted last so the /api routes above take precedence.
    app.mount(
        "/",
        StaticFiles(directory=str(config.WEB_DIR), html=True),
        name="web",
    )
    return app


app = create_app()


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the API with uvicorn (used by the ``trp serve`` CLI command)."""
    import uvicorn

    if reload:
        uvicorn.run("src.api.app:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
