import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";
import React from "react";
import { setUp } from "./harness.mjs";

const {
  testing: { fireEvent, render, screen, waitFor, within },
  modules: [{ App }],
} = await setUp("http://viewer.test/?dataset=test%2Fexample&version=1.2.3#token=test-secret", ["src/app.tsx"]);

const manifest = {
  metadata: { dataset_id: "test/example", dataset_version: "1.2.3", name: "Example", description: "",
    source_url: null, domains: [], tags: [], citation: null, access: "public", license: "MIT" },
  counts: { records: 1, sources: 1, signals: 1, tasks: {}, annotation_occurrences: 2 },
  schema: { tasks: [], time_series_specs: [], annotations: [] },
};
const annotation = (key, value, facts = {}) => ({
  key, value_type: Array.isArray(value) ? "list" : typeof value, value, span: null, unit: null,
  description: null, source: null, confidence: null, metadata: {}, occurrence_metadata: {}, ...facts,
});
const json = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
let requests, annotations;
beforeEach(() => {
  requests = [];
  annotations = () => json({ annotations: [] });
  globalThis.fetch = async (path, options) => {
    const url = new URL(path, "http://viewer.test");
    requests.push(url);
    assert.equal(options.headers.Authorization, "Bearer test-secret");
    assert.equal(url.searchParams.get("dataset_id"), "test/example");
    assert.equal(url.searchParams.get("version"), "1.2.3");
    switch (url.pathname) {
      case "/api/v1/manifest": return json(manifest);
      case "/api/v1/dataset-annotations": return annotations();
      default: throw new Error("unexpected request " + url.pathname);
    }
  };
});

async function table() {
  render(React.createElement(App));
  return within(await screen.findByRole("table", { name: "Dataset annotations" }));
}

test("dataset annotations show as rows with only the columns some row uses", async () => {
  annotations = () => json({ annotations: [
    annotation("source_url", "https://example.org/release"),
    annotation("age_unit", 64, { unit: "year", description: "Median subject age." }),
  ] });
  const rows = await table();
  assert.deepEqual(rows.getAllByRole("columnheader").map(cell => cell.textContent), ["Key", "Value", "Unit", "Description"]);
  assert.ok(rows.getByText("https://example.org/release"));
  assert.ok(rows.getByText("64"));
  assert.ok(rows.getByText("year"));
  assert.ok(screen.getByRole("heading", { name: "Dataset annotations (2)" }));
});

test("list values collapse to a count and preview, and expand to every item", async () => {
  annotations = () => json({ annotations: [annotation("vocabulary", ["sleep", "sitting", "walking", "bicycling"])] });
  const rows = await table();
  const summary = rows.getByText("4 items");
  assert.match(summary.closest("summary").textContent, /sleep, sitting, walking, …$/);
  assert.equal(rows.getAllByRole("listitem").length, 4);
  assert.ok(rows.getByText("bicycling"));
});

test("metadata merges content and occurrence facts behind a disclosure", async () => {
  annotations = () => json({ annotations: [
    annotation("plain", "x"),
    annotation("described", "y", { metadata: { origin: "codebook" }, occurrence_metadata: { row: 3 } }),
  ] });
  const rows = await table();
  assert.ok(rows.getByRole("columnheader", { name: "Metadata" }));
  const [, , described] = rows.getAllByRole("row");
  assert.deepEqual(JSON.parse(within(described).getByText(/codebook/).textContent), { origin: "codebook", row: 3 });
});

test("a dataset without annotations says so", async () => {
  render(React.createElement(App));
  assert.ok(await screen.findByText("This dataset has no dataset-level annotations."));
  assert.equal(screen.queryByRole("table", { name: "Dataset annotations" }), null);
});

test("a failed request shows its message and retries", async () => {
  let attempts = 0;
  annotations = () => ++attempts === 1
    ? json({ detail: { code: "operation_limit", message: "Attached annotations exceeds the configured limit." } }, 413)
    : json({ annotations: [annotation("source_url", "https://example.org")] });
  render(React.createElement(App));
  assert.ok(await screen.findByText("Attached annotations exceeds the configured limit."));
  fireEvent.click(screen.getByRole("button", { name: "Retry annotations" }));
  assert.ok(await screen.findByText("https://example.org"));
  await waitFor(() => assert.equal(requests.filter(url => url.pathname === "/api/v1/dataset-annotations").length, 2));
});
