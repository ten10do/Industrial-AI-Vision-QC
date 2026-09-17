import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// The dashboard talks to the backend through a Vite dev proxy so the browser
// only ever calls same-origin paths (/api/v1/...). The backend base URL is
// injected via the VITE_BACKEND_URL env var (default http://127.0.0.1:8000).
const backend = process.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: backend, changeOrigin: true, ws: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
  },
});
