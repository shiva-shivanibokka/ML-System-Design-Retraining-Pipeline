#!/usr/bin/env bash
# Push the serving app to a Hugging Face Docker Space.
# Requires: HF_TOKEN (write), HF_SPACE_ID like "username/credit-risk-model-api".
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN}"
: "${HF_SPACE_ID:?set HF_SPACE_ID as user/space}"

WORK="$(mktemp -d)"
git clone "https://user:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE_ID}" "$WORK"

# Copy the app + the Space README header (README_SPACE.md -> README.md).
rsync -a --delete \
  --exclude ".git" --exclude "frontend" --exclude "tests" --exclude "docs" \
  --exclude "data" --exclude "mlruns" --exclude "node_modules" \
  ./ "$WORK/"
cp serving/Dockerfile "$WORK/Dockerfile"
cp serving/README_SPACE.md "$WORK/README.md"

cd "$WORK"
git add -A
git commit -m "deploy: sync serving app" || echo "no changes"
git push
echo "Pushed to Space ${HF_SPACE_ID}. Build logs: https://huggingface.co/spaces/${HF_SPACE_ID}"
