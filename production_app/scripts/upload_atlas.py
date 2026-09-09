"""Upload the generated atlas dataset to the deployed stack's artifact bucket."""

import json
import subprocess
from pathlib import Path


STACK = "som-demo"
REGION = "ap-southeast-2"
LOCAL_PATH = Path.home() / ".cache" / "som-atlas" / "embeddings.npz"
S3_KEY = "atlas/embeddings.npz"


def main():
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
    outputs = {item["OutputKey"]: item["OutputValue"] for item in json.loads(raw)}
    destination = f"s3://{outputs['ArtifactBucketName']}/{S3_KEY}"
    subprocess.run(
        ["aws", "s3", "cp", str(LOCAL_PATH), destination, "--region", REGION],
        check=True,
    )
    print(f"uploaded {destination}")


if __name__ == "__main__":
    main()
