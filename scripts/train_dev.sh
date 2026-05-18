set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.training.train --epochs 5 --batch-size 64 --n-samples 512 --out-dir runs/dev
