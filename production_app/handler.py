import json
import os
import uuid
from io import BytesIO

import boto3
import numpy as np

from kohonen.mosaic import atlas_to_png_bytes
from kohonen.png import weights_to_png_bytes
from kohonen.som import SOM

BUCKET = os.environ["BUCKET"]
ATLAS_KEY = os.environ.get("ATLAS_KEY", "atlas/embeddings.npz")
LIMITS = {
    "width": (3, 80),
    "height": (3, 80),
    "n_iter": (1, 400),
    "n_samples": (2, 50),
    "lr0": (1e-4, 1.0),
}
ATLAS_LIMITS = {
    "width": (8, 20),
    "height": (8, 20),
    "n_iter": (25, 150),
}

s3 = boto3.client("s3")
atlas_data = None


def _json_body(event):
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _clamp_int(body, key, default, limits=LIMITS):
    lo, hi = limits[key]
    v = int(body.get(key, default))
    return max(lo, min(hi, v))


def _clamp_float(body, key, default):
    lo, hi = LIMITS[key]
    v = float(body.get(key, default))
    return max(lo, min(hi, v))


def _response(status, payload):
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json",
            "access-control-allow-origin": "*",
            "access-control-allow-headers": "authorization,content-type",
        },
        "body": json.dumps(payload),
    }


def lambda_handler(event, context):
    try:
        path = event.get("rawPath", "")
        if path.endswith("/init-atlas"):
            return _init_atlas(event)
        if path.endswith("/train-atlas"):
            return _train_atlas(event)
        if path.endswith("/init"):
            return _init(event)
        return _train(event)
    except Exception as e:
        return _response(400, {"message": str(e)})


def _put_png(png, filename, extra):
    run_id = str(uuid.uuid4())
    key = f"runs/{run_id}/{filename}"
    s3.put_object(Bucket=BUCKET, Key=key, Body=png, ContentType="image/png")
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=900,
    )
    extra = {"run_id": run_id, "image_url": url, **extra}
    return _response(200, extra)


def _train(event):
    body = _json_body(event)

    width = _clamp_int(body, "width", 10)
    height = _clamp_int(body, "height", 10)
    n_iter = _clamp_int(body, "n_iter", 100)
    n_samples = _clamp_int(body, "n_samples", 10)
    lr0 = _clamp_float(body, "lr0", 0.1)
    seed = int(body.get("seed", 7))
    radius_mask = bool(body.get("radius_mask", False))

    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, 3))

    som = SOM(width, height, n_iter, lr0=lr0, seed=seed, radius_mask=radius_mask)
    som.train(X)
    png = weights_to_png_bytes(som.weights)

    return _put_png(
        png,
        "som.png",
        {
            "quantization_error": som.quantization_error(X),
            "width": width,
            "height": height,
            "n_iter": n_iter,
            "n_samples": n_samples,
            "lr0": lr0,
            "seed": seed,
            "radius_mask": radius_mask,
        },
    )


def _init(event):
    body = _json_body(event)
    width = _clamp_int(body, "width", 10)
    height = _clamp_int(body, "height", 10)
    seed = int(body.get("seed", 7))

    som = SOM(width, height, n_iter=2, seed=seed)
    som.init_weights(3)
    png = weights_to_png_bytes(som.weights)

    return _put_png(
        png,
        "som-init.png",
        {"width": width, "height": height, "seed": seed},
    )


def _load_atlas():
    global atlas_data
    if atlas_data is None:
        obj = s3.get_object(Bucket=BUCKET, Key=ATLAS_KEY)
        with np.load(BytesIO(obj["Body"].read()), allow_pickle=False) as data:
            atlas_data = {
                "X": data["X"],
                "thumbs": data["thumbs"],
                "encoder": str(data["encoder"]),
                "class_names": data["class_names"].tolist(),
            }
    return atlas_data


def _train_atlas(event):
    body = _json_body(event)
    width = _clamp_int(body, "width", 16, ATLAS_LIMITS)
    height = _clamp_int(body, "height", 16, ATLAS_LIMITS)
    n_iter = _clamp_int(body, "n_iter", 100, ATLAS_LIMITS)
    seed = int(body.get("seed", 7))

    data = _load_atlas()
    X = data["X"]
    som = SOM(width, height, n_iter, seed=seed).train(X)
    png = atlas_to_png_bytes(som.weights, X, data["thumbs"])

    return _put_png(
        png,
        "atlas.png",
        {
            "quantization_error": som.quantization_error(X),
            "width": width,
            "height": height,
            "n_iter": n_iter,
            "n_images": len(X),
            "embedding_dim": X.shape[1],
            "encoder": data["encoder"],
            "classes": data["class_names"],
        },
    )


def _init_atlas(event):
    body = _json_body(event)
    width = _clamp_int(body, "width", 16, ATLAS_LIMITS)
    height = _clamp_int(body, "height", 16, ATLAS_LIMITS)
    seed = int(body.get("seed", 7))

    data = _load_atlas()
    X = data["X"]
    som = SOM(width, height, n_iter=2, seed=seed)
    som.init_weights(X.shape[1])
    png = atlas_to_png_bytes(som.weights, X, data["thumbs"])

    return _put_png(
        png,
        "atlas-init.png",
        {
            "width": width,
            "height": height,
            "seed": seed,
            "n_images": len(X),
            "embedding_dim": X.shape[1],
            "encoder": data["encoder"],
            "classes": data["class_names"],
        },
    )
