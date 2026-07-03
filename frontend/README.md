# Credit Risk Pipeline — Frontend

A Next.js 14 (App Router) dashboard that mirrors the retired Streamlit pages: pipeline
overview, drift monitor, training history, model registry, slice performance, and model
cards. It is a **pure client** of the M4 serving API — no server-side state of its own.

## Local development

```bash
cd frontend
npm install
cp .env.example .env.local   # then set NEXT_PUBLIC_API_URL
npm run dev                  # http://localhost:3000
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | No (falls back to empty-state UI) | Base URL of the serving API (HF Space), no trailing slash — e.g. `https://<user>-credit-risk-model-api.hf.space` |

With `NEXT_PUBLIC_API_URL` unset, every page still builds and renders — the typed API
client (`lib/api.ts`) short-circuits to fallback/empty data instead of making a network
call, so pages show their empty-state UI rather than erroring.

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Start the dev server |
| `npm run build` | Production build (used in CI) |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run lint` | `next lint` |

## Deploying to Vercel

1. Go to [vercel.com/new](https://vercel.com/new) and import this GitHub repository.
2. Set **Root Directory** to `frontend`.
3. Add the environment variable `NEXT_PUBLIC_API_URL`, pointing at the deployed HF Space
   serving API URL (see `.env.example`).
4. Deploy. Vercel auto-detects Next.js — no custom build command or output directory is
   needed.

Every push to `main` redeploys automatically once the Vercel project is connected.
