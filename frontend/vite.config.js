import { defineConfig } from "vite";

export default defineConfig({
  esbuild: { jsx: "automatic" },
  build: {
    target: "es2020",
    outDir: "../packages/timenet/src/timenet/viewer/static",
    emptyOutDir: false,
    rollupOptions: { output: {
      entryFileNames: "app.js",
      assetFileNames: "app.[ext]",
    } },
  },
  // The server exposes packaged assets under /static, and the shell at /.
  base: "/static/",
});
