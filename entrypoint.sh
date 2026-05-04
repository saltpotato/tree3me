#!/usr/bin/env bash
set -e

export REPO_ZIP_URL="${REPO_ZIP_URL:-https://github.com/saltpotato/tree3explorer/archive/refs/heads/main.zip}"
export APP_DIR="${APP_DIR:-/workspace/tree3explorer}"
export RUN_CMD="${RUN_CMD:-python train_frontier_actorcritic.py}"

python - <<'PY'
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

url = os.environ["REPO_ZIP_URL"]
app_dir = Path(os.environ["APP_DIR"])
zip_path = Path("/workspace/repo.zip")
extract_dir = Path("/workspace/repo_extract")

if app_dir.exists():
    shutil.rmtree(app_dir)
if extract_dir.exists():
    shutil.rmtree(extract_dir)

print(f"Downloading {url}", flush=True)
urllib.request.urlretrieve(url, zip_path)

extract_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(extract_dir)

root = next(extract_dir.iterdir())
shutil.move(str(root), str(app_dir))
print(f"Repository extracted to {app_dir}", flush=True)
PY

cd "$APP_DIR"
mkdir -p models outputs

echo "Running: $RUN_CMD"
exec bash -lc "$RUN_CMD"