import json
import os
import uuid

import boto3
import numpy as np

from kohonen.png import weights_to_png_bytes
from kohonen.som import SOM

BUCKET = os.environ["BUCKET"]
LIMITS = {
    "width": (3, 80),
    "height": (3, 80),
    "n_iter": (1, 400),
    "n_samples": (2, 50),
    "lr0": (1e-4, 1.0),
}

s3 = boto3.client("s3")


def _json_body(event):
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _clamp_int(body, key, default):
    lo, hi = LIMITS[key]
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
        return _train(event)
    except Exception as e:
        return _response(400, {"message": str(e)})


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
    qe = som.quantization_error(X)

    run_id = str(uuid.uuid4())
    key = f"runs/{run_id}/som.png"
    s3.put_object(Bucket=BUCKET, Key=key, Body=png, ContentType="image/png")
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=900,
    )

    return _response(
        200,
        {
            "run_id": run_id,
            "quantization_error": qe,
            "image_url": url,
            "width": width,
            "height": height,
            "n_iter": n_iter,
            "n_samples": n_samples,
            "lr0": lr0,
            "seed": seed,
            "radius_mask": radius_mask,
        },
    )
