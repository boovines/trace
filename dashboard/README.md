# dashboard — usage analytics UI

Vite + React 18 + TypeScript. Renders the daily-usage statistics produced by
the recorder service (`/stats/summary` on the gateway).

## Develop

```bash
# 1. start the gateway (Python)
uv run uvicorn gateway.main:app --port 8765

# 2. in another shell, start vite
cd dashboard
pnpm install
pnpm dev      # http://127.0.0.1:5173
```

Vite proxies `/stats/*`, `/trajectories/*`, and `/recorder/*` to the gateway,
so the React app uses the same relative URLs in dev and prod.

## Build for production

```bash
cd dashboard
pnpm build    # emits dashboard/dist/
```

The gateway serves `dashboard/dist/` as static files at `GET /dashboard/`
when the directory exists.
