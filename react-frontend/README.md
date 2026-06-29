# RTG React Frontend

A React rebuild of the Flutter `rtg-flutter-app`, talking to the same FastAPI
backend (`backend/`). Full feature parity, tuned for a snappy feel.

## Run

```bash
# 1. start the backend (from repo root)
python -m backend.run_server          # serves http://127.0.0.1:8000

# 2. start this app
cd react-frontend
npm install
npm run dev                           # http://localhost:5173
```

Backend URL is set in `.env` (`VITE_API_URL`, default `http://127.0.0.1:8000`).

## Stack & why

- **Vite + React 18 + TypeScript** — instant HMR, typed API surface.
- **Zustand** (`store.ts`, `chat.ts`, `ui.ts`) — slice subscriptions, so a 2s
  telemetry tick never re-renders the whole app (no Context re-render storms).
- **uPlot** (`Chart.tsx`) — live reward/loss curve at 60fps.
- **Vanilla CSS variables** (`theme.css`) — 8 accents + dark/light, zero runtime.

## Performance notes

- **Viewport** (`Viewport.tsx`): WS binary frames (JPEG **or** RTGF-raw) decode
  off-thread via `createImageBitmap` and draw straight to `<canvas>` — React
  never re-renders per frame, only on the ~1Hz status text. In-flight frames are
  dropped if a decode is still running.
- **Chat** (`chat.ts`): streamed tokens buffer and flush every 66ms instead of
  re-rendering per token; inline `<think>` blocks are parsed out into a
  collapsible section.

## Layout

`store.ts` mirrors the Flutter `AppState` (same fields, polling cadence, derived
getters). Panels live in `src/panels/` (Home, Robot, ObsAction, Rewards,
Training, Evaluation, Settings, Logs); shared pieces in `src/components/`.

## Skipped (add when needed)

- **Markdown rendering** in chat bubbles — plain text with preserved whitespace.
  Add a renderer if rich formatting matters.
- **File picker** in Robot Setup — a text input takes a server-side URDF path
  (the backend loads by path; a browser can't read the server filesystem).
- **Run compare** uses the already-loaded run summaries client-side rather than a
  dedicated `/runs/compare` call.
