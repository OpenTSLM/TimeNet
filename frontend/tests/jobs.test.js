import { readFile } from "node:fs/promises";
import { test } from "node:test";
import assert from "node:assert/strict";
import ts from "typescript";

// Exercise the production TypeScript lifecycle without a browser or a second bundler.
async function load(name) {
  const source = await readFile(new URL(`../src/${name}.ts`, import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } });
  return import(`data:text/javascript,${encodeURIComponent(outputText)}`);
}
const { JobRun } = await load("jobs");
const { progressLabel } = await load("api");

test("disposal cancels a job whose admission response arrives later", async () => {
  let admit;
  const cancelled = [];
  const client = { request: () => new Promise((resolve) => { admit = resolve; }), cancel: async (id) => cancelled.push(id) };
  const operation = new JobRun(client);
  const result = operation.start("/windows", {});
  operation.dispose();
  admit({ job_id: "late" });
  await assert.rejects(result, { name: "AbortError" });
  assert.deepEqual(cancelled, ["late"]);
});

test("disposal aborts polling and cancels the owned job", async () => {
  let polled;
  const entered = new Promise((resolve) => { polled = resolve; });
  const cancelled = [];
  const client = { request: async (path, _body, signal) => {
    if (path === "/windows") return { job_id: "active" };
    polled();
    return new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(new DOMException("cancelled", "AbortError"))));
  }, cancel: async (id) => cancelled.push(id) };
  const operation = new JobRun(client);
  const result = operation.start("/windows", {});
  await entered;
  operation.dispose();
  await assert.rejects(result, { name: "AbortError" });
  assert.deepEqual(cancelled, ["active"]);
});

test("progress labels expose units and distinguish unknown from zero totals", () => {
  assert.equal(progressLabel({ state: "running", progress: { completed: "20", total: null, phase: "scanning", unit: "component steps" } }),
    "running: scanning — 20 total unknown component steps");
  assert.match(progressLabel({ state: "succeeded", progress: { completed: "0", total: "0", unit: "datasets" } }), /0 of 0 datasets/);
});
