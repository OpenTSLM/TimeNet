import { defineConfig } from "vite";

export default defineConfig({
  base: "/static/",
  build: {
    target: "es2020",
    outDir: "../packages/timenet/src/timenet/viewer/static",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: "app.js",
        assetFileNames: "app.[ext]",
      },
    },
  },
});
