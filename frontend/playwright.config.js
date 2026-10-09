import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "tests",
  testMatch: "**/*.spec.js",
  use: { baseURL: "http://127.0.0.1:8765", headless: true },
  webServer: {
    command: "uv run --no-sync python tests/server.py",
    url: "http://127.0.0.1:8765",
    reuseExistingServer: false,
    timeout: 60000,
  },
});
