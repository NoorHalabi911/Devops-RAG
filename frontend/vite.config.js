import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    strictPort: true,
    proxy: {
      // Dev-only: avoid browser CORS / localhost mapping issues.
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});

