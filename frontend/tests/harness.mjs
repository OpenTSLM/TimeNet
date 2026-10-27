import { after, afterEach } from "node:test";
import { JSDOM } from "jsdom";
import { build } from "esbuild";
import { mkdtemp, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

// Install a jsdom window at `url`, then bundle and import `entryPoints` against it. The app reads
// its token from the launch URL when its module loads, so the window must exist first.
export async function setUp(url, entryPoints) {
  const dom = new JSDOM("<!doctype html><html><body></body></html>", { url });
  for (const name of ["window", "document", "location", "history", "sessionStorage", "HTMLElement", "Node"]) {
    Object.defineProperty(globalThis, name, { value: dom.window[name], configurable: true });
  }
  globalThis.addEventListener = dom.window.addEventListener.bind(dom.window);
  globalThis.removeEventListener = dom.window.removeEventListener.bind(dom.window);
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const testing = await import("@testing-library/react");
  const bundle = await mkdtemp(join(tmpdir(), "timenet-frontend-tests-"));
  await symlink(resolve("node_modules"), join(bundle, "node_modules"), "dir");
  await build({
    entryPoints, bundle: true, platform: "node",
    format: "esm", packages: "external", outdir: bundle, jsx: "automatic",
  });
  const modules = await Promise.all(entryPoints.map(
    entry => import(pathToFileURL(join(bundle, basename(entry).replace(/\.tsx?$/, ".js")))),
  ));
  after(async () => { dom.window.close(); await rm(bundle, { recursive: true, force: true }); });
  afterEach(testing.cleanup);
  return { testing, modules };
}
