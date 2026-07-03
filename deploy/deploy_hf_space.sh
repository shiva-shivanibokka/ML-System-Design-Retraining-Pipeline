#!/usr/bin/env bash
# Push the serving app to a Hugging Face Docker Space.
# Requires: HF_TOKEN (write), HF_SPACE_ID like "username/credit-risk-model-api".
# Portable: uses `git archive` (no rsync) so it runs on Windows Git Bash + Linux CI.
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN}"
: "${HF_SPACE_ID:?set HF_SPACE_ID as user/space}"

WORK="$(mktemp -d)"
git clone "https://user:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE_ID}" "$WORK"

# Export the repo's tracked files at HEAD into the Space clone. `git archive`
# naturally excludes gitignored content (.env, data/, mlruns/, node_modules/).
git archive HEAD | tar -x -C "$WORK"

# Slim the serving image: the API needs none of these.
rm -rf "$WORK/frontend" "$WORK/tests" "$WORK/docs" "$WORK/streamlit_app" 2>/dev/null || true

# HF Space config lives at the repo root: the Dockerfile + a README with the
# required YAML front-matter (sdk: docker, app_port: 7860).
cp serving/Dockerfile "$WORK/Dockerfile"
cp serving/README_SPACE.md "$WORK/README.md"

cd "$WORK"
git add -A
git commit -m "deploy: sync serving app" || echo "no changes"
git push
echo "Pushed to Space ${HF_SPACE_ID}. Build logs: https://huggingface.co/spaces/${HF_SPACE_ID}"
