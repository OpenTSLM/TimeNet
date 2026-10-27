import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";
import React from "react";
import { setUp } from "./harness.mjs";

const {
  testing: { act, cleanup, fireEvent, render, screen, waitFor, within },
  modules: [{ App }, navigation],
} = await setUp(
  "http://viewer.test/?dataset=test%2Fexample&version=1.2.3&view=tasks&task=task-a#token=test-secret",
  ["src/app.tsx", "src/navigation.ts"],
);

const identity = { dataset: "test/example", version: "1.2.3", view: "tasks", task: "task-a", split: "train", offset: "20" };
const task = (id = "task-a", inputs = ["record-a", "record-b"]) => ({
  id, task_type: "answer", split: "train", prompt: "Prompt for " + id, rationale: "Task rationale",
  input_record_ids: inputs, parent_task_ids: [], input_modalities: [], scope: null,
  targets: [{ kind: "record", id: "target-record" }],
  configuration: { candidate_record_ids: ["candidate-record"] }, metadata: {},
  annotations: [], input_annotations: [], target_annotations: [],
});
const signal = (id, source_id) => ({
  id, source_id, name: "Signal " + id, n_values: 8, metadata: {}, annotations: [],
  spec: { dtype: "float64", unit_value: "mV" }, axis: { kind: "ordinal" },
  plot_supported: true, plot_reason: null,
});
const record = id => ({
  id, start_time: null, time_span: null, metadata: { subject: id }, annotations: [],
  sources: [
    { id: "source-a", name: "Monitor", parent_source_id: null, metadata: { device: "monitor" }, annotations: [] },
    { id: "child-a", name: "Child monitor", parent_source_id: "source-a", metadata: {}, annotations: [] },
    { id: "sibling-a", name: "Other monitor", parent_source_id: null, metadata: {}, annotations: [] },
  ],
  signals: [signal("signal-a", "source-a"), signal("child-signal", "child-a"), signal("other-signal", "sibling-a")],
});
const manifest = {
  metadata: { dataset_id: identity.dataset, dataset_version: identity.version, name: "Example", description: "",
    source_url: null, domains: [], tags: [], citation: null, access: "public", license: "MIT" },
  counts: { records: 4, sources: 3, signals: 3, tasks: { answer: 2 }, annotation_occurrences: 0 },
  schema: { tasks: [], time_series_specs: [], annotations: [] },
};
const json = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
let requests, override;
beforeEach(() => {
  requests = [];
  override = () => undefined;
  globalThis.fetch = async (path, options) => {
    const url = new URL(path, "http://viewer.test");
    const request = { path: url.pathname, params: url.searchParams, signal: options.signal };
    requests.push(request);
    assert.equal(options.headers.Authorization, "Bearer test-secret");
    assert.equal(url.searchParams.get("dataset_id"), identity.dataset);
    assert.equal(url.searchParams.get("version"), identity.version);
    const custom = override(request);
    if (custom !== undefined) return custom;
    switch (request.path) {
      case "/api/v1/manifest": return json(manifest);
      case "/api/v1/task": {
        const id = url.searchParams.get("task_id");
        return json(task(id, id === "task-a" ? undefined : ["record-b"]));
      }
      case "/api/v1/tasks": return json({ items: [{ id: "task-b", task_type: "answer", split: "train", prompt_preview: "Other task" }],
        offset: 20, limit: 20, has_more: false });
      case "/api/v1/record": return json(record(url.searchParams.get("record_id")));
      case "/api/v1/signal-window": {
        const start = Number(url.searchParams.get("start")), stop = Number(url.searchParams.get("stop"));
        return json({ start, stop, values: Array.from({ length: stop - start }, (_, i) => i), nonfinite_steps: [] });
      }
      default: throw new Error("Unexpected endpoint: " + request.path);
    }
  };
});
function mount(changes = {}) {
  cleanup();
  history.replaceState(null, "", "/?" + new URLSearchParams({ ...identity, ...changes }));
  const result = render(React.createElement(App));
  return result;
}
const current = () => new URLSearchParams(location.search);
const calls = path => requests.filter(request => request.path === "/api/v1/" + path);
const inspector = () => screen.getByRole("region", { name: "Record inspector" });
const selector = () => screen.getByRole("complementary", { name: "Task records" });
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};
async function loaded(id = "record-a") {
  await waitFor(() => assert.ok(screen.getByRole("heading", { name: id })));
  await waitFor(() => assert.ok(within(inspector()).queryByRole("status") === null));
}
async function selectSignal() {
  fireEvent.click(within(inspector()).getByRole("button", { name: "Signal Signal signal-a (signal-a)" }));
  await screen.findByRole("button", { name: "Load window" });
}

test("task embeds one inspector and selecting a signal loads its initial window", async () => {
  mount();
  await loaded();
  assert.ok(screen.getByText("Prompt for task-a"));
  assert.equal(screen.getAllByRole("region", { name: "Record inspector" }).length, 1);
  const hierarchy = screen.getByRole("navigation", { name: "Record hierarchy" });
  assert.equal(within(hierarchy).queryAllByRole("heading").length, 0);
  assert.ok(within(hierarchy).queryByText("record-a") === null);
  assert.deepEqual(calls("record").map(request => request.params.get("record_id")), ["record-a"]);
  assert.equal(calls("signal-window").length, 0);
  assert.ok(screen.getByRole("heading", { name: "Select a signal" }));
  await selectSignal();
  await screen.findByRole("img", { name: /Signal values from step 0 to 7/ });
  assert.equal(calls("signal-window").length, 1);
  const request = calls("signal-window").at(-1).params;
  assert.equal(request.get("record_id"), "record-a");
  assert.equal(request.get("signal_id"), "signal-a");
  assert.equal(request.get("start"), "0");
  assert.equal(request.get("stop"), "8");
  assert.equal(current().get("view"), "tasks");
  assert.ok(screen.getByText("Prompt for task-a"));
});

test("record selection replaces the inspector, clears child state, and handles target and candidate references", async () => {
  mount({ record: "record-a", source: "source-a", signal: "signal-a", start: "2", stop: "5" });
  await loaded();
  fireEvent.click(within(selector()).getByRole("button", { name: "record-b Input" }));
  await loaded("record-b");
  assert.equal(screen.getAllByRole("region", { name: "Record inspector" }).length, 1);
  for (const key of ["source", "signal", "start", "stop"]) assert.equal(current().has(key), false);
  for (const id of ["target-record", "candidate-record"]) {
    fireEvent.click(screen.getByRole("button", { name: id, exact: true }));
    await loaded(id);
    assert.equal(within(selector()).getByRole("button", { name: new RegExp("^" + id + " ") }).getAttribute("aria-pressed"), "true");
  }
  assert.deepEqual(calls("record").map(request => request.params.get("record_id")),
    ["record-a", "record-b", "target-record", "candidate-record"]);
});

test("one record action preserves selection and returns to its task", async () => {
  mount({ record: "record-a", source: "source-a", signal: "signal-a", start: "2", stop: "5" });
  await loaded();
  const link = screen.getByRole("link", { name: "Open record record-a in record view" });
  assert.equal(link.target, "_blank");
  assert.equal(link.rel, "opener");
  const params = new URL(link.href).searchParams;
  assert.equal(params.get("view"), "records");
  assert.equal(params.get("task"), "task-a");
  assert.equal(params.get("signal"), "signal-a");
  mount(Object.fromEntries(params));
  await loaded();
  assert.ok(within(inspector()).getByLabelText("Source Other monitor (sibling-a)"));
  assert.ok(screen.queryByRole("link", { name: /Open record/ }) === null);
  assert.ok(screen.queryByRole("link", { name: /Open source/ }) === null);
  fireEvent.click(screen.getByRole("link", { name: /Return to task/ }));
  await loaded();
  for (const key of ["record", "source", "signal", "start", "stop", "split", "offset"])
    assert.equal(current().get(key), params.get(key));
  assert.ok(screen.getByText("Prompt for task-a"));
  assert.equal(screen.getByLabelText("Start step").value, "2");
  assert.equal(screen.getByLabelText("Stop step (exclusive)").value, "5");
});

test("missing record children never request values", async () => {
  mount({ view: "records", task: "", record: "record-a", source: "missing" });
  await screen.findByText("Source not found in this record.");
  assert.equal(calls("signal-window").length, 0);
  mount({ view: "records", task: "", record: "record-a", source: "child-a", signal: "missing", start: "0", stop: "4" });
  await screen.findByText("Signal not found in this record.");
  assert.equal(calls("signal-window").length, 0);
});

test("task changes clear stale inspector state and select the new task's first input", async () => {
  mount({ record: "record-a", source: "source-a", signal: "signal-a", start: "2", stop: "5" });
  await loaded();
  fireEvent.click(screen.getByRole("button", { name: "← Task list" }));
  await screen.findByRole("button", { name: /task-b/ });
  assert.equal(current().get("split"), "train");
  assert.equal(current().get("offset"), "20");
  for (const key of ["record", "source", "signal", "start", "stop"]) assert.equal(current().has(key), false);
  fireEvent.click(screen.getByRole("button", { name: /task-b/ }));
  await loaded("record-b");
  assert.ok(screen.getByText("Prompt for task-b"));
  assert.equal(current().get("record"), "record-b");
});

test("reload, back, and forward restore the selected record", async () => {
  mount();
  await loaded();
  fireEvent.click(within(selector()).getByRole("button", { name: "record-b Input" }));
  await loaded("record-b");
  history.back();
  await waitFor(() => assert.equal(current().get("record"), "record-a"));
  await loaded();
  history.forward();
  await waitFor(() => assert.equal(current().get("record"), "record-b"));
  await loaded("record-b");
  mount(Object.fromEntries(current()));
  await loaded("record-b");
});

test("switching records aborts metadata requests and ignores late responses", async () => {
  const pending = deferred();
  override = request => request.path === "/api/v1/record" && request.params.get("record_id") === "record-a"
    ? pending.promise : undefined;
  mount();
  await waitFor(() => assert.equal(calls("record").length, 1));
  const request = calls("record")[0];
  fireEvent.click(within(selector()).getByRole("button", { name: "record-b Input" }));
  await loaded("record-b");
  assert.equal(request.signal.aborted, true);
  await act(async () => pending.resolve(json(record("record-a"))));
  assert.ok(within(inspector()).getByRole("heading", { name: "record-b" }));
});

test("switching records aborts a submitted window and ignores late values", async () => {
  const pending = deferred();
  override = request => request.path === "/api/v1/signal-window" ? pending.promise : undefined;
  mount({ record: "record-a", source: "source-a", signal: "signal-a", start: "2", stop: "5" });
  await waitFor(() => assert.equal(calls("signal-window").length, 1));
  const request = calls("signal-window")[0];
  fireEvent.click(within(selector()).getByRole("button", { name: "record-b Input" }));
  await loaded("record-b");
  assert.equal(request.signal.aborted, true);
  await act(async () => pending.resolve(json({ start: 2, stop: 5, values: [1, 2, 3], nonfinite_steps: [] })));
  assert.ok(screen.queryByRole("img") === null);
});

test("late task responses cannot replace a newly selected task", async () => {
  const pending = deferred();
  override = request => request.path === "/api/v1/task" && request.params.get("task_id") === "task-a"
    ? pending.promise : undefined;
  mount();
  await waitFor(() => assert.equal(calls("task").length, 1));
  const request = calls("task")[0];
  fireEvent.click(screen.getByRole("button", { name: "← Task list" }));
  fireEvent.click(await screen.findByRole("button", { name: /task-b/ }));
  await loaded("record-b");
  assert.equal(request.signal.aborted, true);
  await act(async () => pending.resolve(json(task())));
  assert.ok(screen.getByText("Prompt for task-b"));
  assert.deepEqual(calls("record").map(request => request.params.get("record_id")), ["record-b"]);
});

test("recoverable window failures retain task context, controls, and retry", async () => {
  let attempts = 0;
  override = request => request.path === "/api/v1/signal-window" && attempts++ === 0
    ? json({ detail: { message: "Try this window again." } }, 503) : undefined;
  mount({ record: "record-a", source: "source-a", signal: "signal-a", start: "2", stop: "5" });
  await screen.findByText("Try this window again.");
  assert.ok(screen.getByRole("button", { name: "Load window" }));
  assert.ok(screen.getByText("Prompt for task-a"));
  fireEvent.click(screen.getByRole("button", { name: "Retry window" }));
  await screen.findByRole("img", { name: /Signal values from step 2 to 4/ });
  assert.equal(calls("signal-window").length, 2);
});

test("tasks without inputs have an empty workspace and do not fetch records", async () => {
  override = request => request.path === "/api/v1/task" ? json({ ...task("empty", []), targets: [], configuration: {} }) : undefined;
  mount({ task: "empty" });
  await screen.findByText("No input records.");
  assert.ok(screen.queryByRole("region", { name: "Record inspector" }) === null);
  assert.equal(calls("record").length, 0);
});

test("restored record IDs outside the task are reset without fetching unrelated records", async () => {
  mount({ record: "unrelated", source: "old-source", signal: "old-signal", start: "0", stop: "4" });
  await loaded();
  assert.equal(current().get("record"), "record-a");
  for (const key of ["source", "signal", "start", "stop"]) assert.equal(current().has(key), false);
  assert.deepEqual(calls("record").map(request => request.params.get("record_id")), ["record-a"]);
  assert.equal(calls("signal-window").length, 0);
});

test("record reference roles are deduplicated and record changes preserve only task context", () => {
  assert.deepEqual(navigation.recordChoices(["a", "b"], ["a"], ["b", "c", "c"]), [
    { id: "a", roles: ["Input", "Target"] }, { id: "b", roles: ["Input", "Candidate"] }, { id: "c", roles: ["Candidate"] },
  ]);
  const next = navigation.selectRecord("?dataset=x%2Fy&version=1.0.0&task=t&split=train&offset=20&source=s&signal=v&start=0&stop=2&token=secret", "b");
  for (const key of ["source", "signal", "start", "stop", "token"]) assert.equal(next.has(key), false);
  assert.equal(next.get("record"), "b");
  assert.equal(next.get("task"), "t");
  assert.equal(next.get("split"), "train");
  assert.equal(next.get("offset"), "20");
});

test("hierarchy labels branches and leaves, highlights the signal, and shows its full path", async () => {
  mount({ record: "record-a", source: "child-a", signal: "child-signal" });
  await loaded();
  const hierarchy = screen.getByRole("navigation", { name: "Record hierarchy" });
  const child = within(hierarchy).getByLabelText("Source Child monitor (child-a)");
  assert.equal(child.closest("details").open, true);
  const leaf = within(hierarchy).getByRole("button", { name: "Signal Signal child-signal (child-signal)" });
  assert.equal(leaf.getAttribute("aria-pressed"), "true");
  assert.ok(within(leaf).getByText("Selected"));
  const path = screen.getByRole("navigation", { name: "Selected signal path" });
  assert.match(path.textContent, /record-a.*Monitor.*Child monitor.*Signal child-signal/);
  assert.equal(calls("signal-window").length, 0);
  fireEvent.click(within(hierarchy).getByRole("button", { name: "Signal Signal signal-a (signal-a)" }));
  await screen.findByRole("heading", { name: "Signal signal-a" });
  assert.equal(current().get("source"), "source-a");
  assert.equal(current().get("signal"), "signal-a");
});

test("selecting a signal bounds its initial window and replaces the previous range", async () => {
  override = request => {
    if (request.path !== "/api/v1/record") return;
    const detail = record("record-a");
    detail.signals[0].n_values = 5000;
    return json(detail);
  };
  mount({ record: "record-a", source: "child-a", signal: "child-signal", start: "2", stop: "5" });
  await loaded();
  await screen.findByRole("img", { name: /Signal values from step 2 to 4/ });
  await selectSignal();
  await screen.findByRole("img", { name: /Signal values from step 0 to 999/ });
  assert.equal(current().get("start"), "0");
  assert.equal(current().get("stop"), "1000");
  fireEvent.change(screen.getByLabelText("Start step"), { target: { value: "10" } });
  fireEvent.change(screen.getByLabelText("Stop step (exclusive)"), { target: { value: "20" } });
  assert.equal(calls("signal-window").length, 2);
  fireEvent.click(screen.getByRole("button", { name: "Load window" }));
  await screen.findByRole("img", { name: /Signal values from step 10 to 19/ });
});

test("selecting empty or unsupported signals does not request values", async () => {
  override = request => {
    if (request.path !== "/api/v1/record") return;
    const detail = record("record-a");
    detail.signals[0].n_values = 0;
    detail.signals[1].plot_supported = false;
    detail.signals[1].plot_reason = "Unsupported dtype";
    return json(detail);
  };
  mount();
  await loaded();
  fireEvent.click(screen.getByRole("button", { name: "Signal Signal signal-a (signal-a)" }));
  await screen.findByText("This signal has no values.");
  fireEvent.click(screen.getByLabelText("Source Child monitor (child-a)"));
  fireEvent.click(screen.getByRole("button", { name: "Signal Signal child-signal (child-signal)" }));
  await screen.findByText("Unsupported dtype");
  assert.equal(calls("signal-window").length, 0);
});

test("plot separates the step index from signal value units", async () => {
  mount();
  await loaded();
  await selectSignal();
  const plot = await screen.findByRole("img", { name: /Signal values from step 0 to 7/ });
  assert.ok(within(plot).getByText("Step index", { exact: true }));
  assert.ok(within(plot).getByText("Value · mV", { exact: true }));
  assert.ok(within(plot).queryByText("Step index · mV") === null);
});

test("restored signals select their owning source and preserve the submitted window", async () => {
  mount({ view: "records", record: "record-a", source: "sibling-a", signal: "child-signal", start: "2", stop: "5" });
  await loaded();
  await screen.findByRole("img", { name: /Signal values from step 2 to 4/ });
  await waitFor(() => assert.equal(current().get("source"), "child-a"));
  assert.equal(screen.getByRole("button", { name: "Inspect source Child monitor" }).getAttribute("aria-pressed"), "true");
  assert.equal(screen.getByRole("button", { name: "Inspect source Other monitor" }).getAttribute("aria-pressed"), "false");
  assert.ok(screen.getByText("Source details: Child monitor"));
  assert.equal(calls("signal-window").length, 1);
});
