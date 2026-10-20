#!/usr/bin/env bash
# One-time setup for the M5 MacBook Air. Run from the project root:
#   bash scripts/setup_mac.sh
set -e
cd "$(dirname "$0")/.."

# Homebrew (skipped if already installed)
if ! command -v brew >/dev/null; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
brew install python@3.12 git node gh || true

python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r app/backend/requirements.txt

python - <<'PY'
import torch
print("torch", torch.__version__, "| Apple GPU (MPS) available:", torch.backends.mps.is_available())
PY
echo "Setup done. Next: source venv/bin/activate && ./scripts/smoke_test.sh"
