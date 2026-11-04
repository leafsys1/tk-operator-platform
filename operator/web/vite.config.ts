import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// 开发期：前端 5173，API 反代到 FastAPI(8077)
// 生产：vite build → web/dist，由 FastAPI 直接托管（单进程部署）
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8077" },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
