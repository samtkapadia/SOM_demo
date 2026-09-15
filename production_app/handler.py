"""
Lambda that the webpage calls *after* login.

User journey (nothing in this file runs when they only open CloudFront):
  1. Browser loads index.html + config.js from CloudFront/S3.
  2. They sign in; the page talks to Cognito and stores a JWT.
  3. Page POSTs here with Authorization: Bearer <jwt>:
       login / Randomize RGB   -> POST /init        -> left RGB image
       Train RGB               -> POST /train       -> right RGB image
       login / Randomize atlas -> POST /init-atlas  -> left photo mosaic
       Train photo atlas       -> POST /train-atlas -> right photo mosaic
  4. Each route builds a PNG, puts it on the artifact S3 bucket, returns a
     15-minute presigned URL. The <img> tags load that URL (S3, not Lambda).

API Gateway's Cognito JWT authorizer runs before this code.
"""

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
# hard caps so one form submit cannot run a huge job on the shared demo account
LIMITS = {
    "width": (3, 100),
    "height": (3, 100),
    "n_iter": (1, 1000),
    "n_samples": (2, 50),
    "lr0": (1e-4, 1.0),
}
ATLAS_LIMITS = {
    "width": (8, 20),
    "height": (8, 20),
    "n_iter": (25, 150),
}

s3 = boto3.client("s3")
atlas_data = None  # filled on first atlas request; kept on warm Lambda environments


def _json_body(event):
    """Turn API Gateway's event['body'] into a dict of form fields (width, seed, …)."""
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _clamp_int(body, key, default, limits=LIMITS):
    """Read an int from the request and clip it into [lo, hi]."""
    lo, hi = limits[key]
    v = int(body.get(key, default))
    return max(lo, min(hi, v))


def _clamp_float(body, key, default):
    """Read a float from the request and clip it into LIMITS[key]."""
    lo, hi = LIMITS[key]
    v = float(body.get(key, default))
    return max(lo, min(hi, v))


def _response(status, payload):
    """JSON back to the browser. CORS * so the CloudFront origin can read it."""
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
    """
    Only entry AWS invokes (template.yaml Handler: handler.lambda_handler).

    event['rawPath'] is the HTTP path after API Gateway already checked the JWT.
    """
    try:
        path = event.get("rawPath", "")
        if path.endswith("/init-atlas"):
            return _init_atlas(event)
        if path.endswith("/train-atlas"):
            return _train_atlas(event)
        if path.endswith("/init"):
            return _init(event)
        # POST /train, and any unmatched path
        return _train(event)
    except Exception as e:
        return _response(400, {"message": str(e)})


def _put_png(png, filename, extra):
    """
    Last step of every route: PNG -> artifact bucket -> JSON {image_url, ...}.

    The browser does not get the PNG bytes in this response; it GETs image_url
    (presigned, 15 min) and that hits S3, not this Lambda.
    """
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
    """
    RGB Train button -> right-hand colour grid.

    Same seed as /init, so this is the organised version of the left-hand map
    (if they did not change width/height/seed in between). Training data is
    n_samples random RGB points, not uploaded files.
    """
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
    """
    After login, and RGB Randomize -> left-hand colour grid.

    No Kohonen updates: just seed -> random (H, W, 3) weights as RGB.
    n_iter on the form is ignored here.
    """
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
    """
    CIFAR embeddings + thumbnails from S3 (uploaded earlier by scripts/upload_atlas.py).

    Cached on this Lambda instance so login's /init-atlas and later Train share one download.
    """
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
    """
    Train photo atlas -> right-hand mosaic.

    Fixed 250 x 128-d embeddings; only grid size / iterations / seed come from the form.
    Each cell shows the nearest training thumbnail (repeats are expected).
    """
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
    """
    After login, and atlas Randomize -> left-hand mosaic.

    Same nearest-image paste as train, but on random 128-d weights (no updates).
    """
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
