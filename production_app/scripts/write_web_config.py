#!/usr/bin/env python3
"""Write web/config.js from a deployed CloudFormation/SAM stack."""

import json
import subprocess
import sys
from pathlib import Path

STACK = sys.argv[1] if len(sys.argv) > 1 else "som-demo"
REGION = sys.argv[2] if len(sys.argv) > 2 else "ap-southeast-2"

raw = subprocess.check_output(
    [
        "aws",
        "cloudformation",
        "describe-stacks",
        "--stack-name",
        STACK,
        "--region",
        REGION,
        "--query",
        "Stacks[0].Outputs",
        "--output",
        "json",
    ],
    text=True,
)
outs = {o["OutputKey"]: o["OutputValue"] for o in json.loads(raw)}
config = {
    "region": REGION,
    "userPoolId": outs["UserPoolId"],
    "clientId": outs["UserPoolClientId"],
    "apiUrl": outs["ApiUrl"].rstrip("/"),
}
dest = Path(__file__).resolve().parents[1] / "web" / "config.js"
dest.write_text(f"window.SOM_CONFIG = {json.dumps(config, indent=2)};\n")
print(f"wrote {dest}")
print(f"site: {outs['SiteUrl']}")
print(f"website bucket: {outs['WebsiteBucketName']}")
print("sync with:")
print(
    f"  aws s3 sync {dest.parent} s3://{outs['WebsiteBucketName']} "
    f"--region {REGION} --exclude config.example.js"
)
