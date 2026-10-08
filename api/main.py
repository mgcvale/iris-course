import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from models import MODELS
from datasets import list_datasets, get_dataset

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GRID_SIZE = 100


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/datasets")
async def datasets():
    return list_datasets()


@app.get("/api/datasets/{key}")
async def dataset(key: str):
    try:
        return get_dataset(key)
    except KeyError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Dataset '{key}' not found")


@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    await ws.accept()
    try:
        data = await ws.receive_text()
        req = json.loads(data)
        points = req.get("points", [])
        model_type = req.get("model_type", "linear")
        model_params = req.get("model_params", {})

        if len(points) < 2:
            await ws.send_json({"type": "error", "msg": "Need at least 2 points"})
            await ws.close()
            return

        num_classes = max(p["color"] for p in points) + 1
        X = np.array([[p["x"], p["y"]] for p in points], dtype=np.float64)
        y = np.array([p["color"] for p in points], dtype=np.int64)

        model_cls = MODELS.get(model_type, MODELS['linear'])
        model = model_cls(**model_params) if model_type == 'svm' else model_cls()
        model.init(num_classes)

        gx, gy = np.meshgrid(
            np.linspace(0, 1, GRID_SIZE),
            np.linspace(0, 1, GRID_SIZE),
        )
        grid_x = gx.ravel()
        grid_y = gy.ravel()

        # epoch counts and send intervals per model
        config = {
            'knn':     dict(epochs=1,   send_every=1,  sleep=0),
            'svm':     dict(epochs=getattr(model, 'epochs', 1), send_every=1, sleep=0.08),
            'genetic': dict(epochs=150, send_every=3,  sleep=0.02),
            'linear':  dict(epochs=200, send_every=5,  sleep=0.02),
        }
        cfg = config.get(model_type, config['linear'])
        epochs       = cfg['epochs']
        send_every   = cfg['send_every']
        sleep_s      = cfg['sleep']

        plateau  = 0

        for epoch in range(epochs):
            loss, acc = model.train_epoch(X, y)

            should_send = (epoch % send_every == 0) or (epoch == epochs - 1)
            if should_send:
                preds, _ = model.predict_grid(grid_x, grid_y)
                await ws.send_json({
                    'type':      'epoch',
                    'epoch':     epoch + 1,
                    'loss':      round(float(loss), 5),
                    'accuracy':  round(float(acc),  5),
                    'grid':      preds.tolist(),
                    'grid_size': GRID_SIZE,
                })
                if sleep_s:
                    await asyncio.sleep(sleep_s)

            # early stop: 100% accuracy held for 10 consecutive sends
            if acc >= 1.0:
                plateau += 1
            else:
                plateau = 0
            if plateau >= 10 and model_type != 'svm':
                break

        await ws.send_json({'type': 'done'})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "front" / "dist"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
