---
title: Credit Risk Model API
emoji: 🏦
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Credit Risk Model Serving API

FastAPI service that serves the champion LightGBM credit-risk model from a
DagsHub-hosted MLflow registry, plus read-only dashboard endpoints consumed by
the Next.js frontend. See `/docs` for the OpenAPI UI.
