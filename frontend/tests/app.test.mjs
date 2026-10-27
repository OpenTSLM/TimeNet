import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { setUp } from "./harness.mjs";

const { testing: { cleanup, render, screen }, modules: [{ App }] } = await setUp(
  "http://viewer.test/?dataset=test%2Fexample&version=1.2.3#token=test-secret", ["src/app.tsx"],
);

function failManifest(response) {
  globalThis.fetch = async (path, options) => {
    assert.equal(options.headers.Authorization, "Bearer test-secret");
    assert.equal(new URL(path, "http://viewer.test").pathname, "/api/v1/manifest");
    return response;
  };
  render(React.createElement(App));
}

test("structured API errors show their message", async () => {
  const detail = { code: "not_found", message: "Dataset version not found.", operation: "manifest", limit: null };
  failManifest({ ok: false, status: 404, json: async () => ({ detail }) });
  assert.ok(await screen.findByText("Dataset version not found."));
});

test("plain string details and security messages show as sent", async () => {
  failManifest({ ok: false, status: 404, json: async () => ({ detail: "Not Found" }) });
  assert.ok(await screen.findByText("Not Found"));
  cleanup();
  failManifest({ ok: false, status: 401, json: async () => ({ message: "authentication required" }) });
  assert.ok(await screen.findByText("authentication required"));
});

test("non-JSON failures fall back to the status without leaking parser errors", async () => {
  failManifest({ ok: false, status: 500, json: async () => { throw new SyntaxError("Unexpected token I"); } });
  assert.ok(await screen.findByText("Viewer request failed (500)."));
  assert.equal(screen.queryByText(/Unexpected token/), null);
});
