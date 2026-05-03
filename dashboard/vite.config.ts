import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `pnpm dev` serves on :5173. Stats requests are proxied to the gateway so
// the React app can `fetch("/stats/summary")` in dev exactly the same way it
// will when served from the gateway in prod.
export default defineConfig({
  // The gateway mounts the built bundle under /dashboard/, so asset URLs in
  // the emitted index.html must be prefixed accordingly. The Vite dev server
  // ignores `base` for HMR, so this doesn't affect `pnpm dev`.
  base: "/dashboard/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/stats": "http://127.0.0.1:8765",
      "/trajectories": "http://127.0.0.1:8765",
      "/recorder": "http://127.0.0.1:8765",
      "/healthz": "http://127.0.0.1:8765",
    },
  },
  build: {
    // Built assets land where the FastAPI gateway mounts them as a static
    // directory (see gateway/main.py). Keeping outDir relative to this
    // project means `pnpm build` "just works" for the gateway too.
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
  },
});
