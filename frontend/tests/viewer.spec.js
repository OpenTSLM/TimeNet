import { test, expect } from "@playwright/test";

test("local assets, authentication, gaps, reduced plots and raw pagination", async ({ page }) => {
  const origins = new Set();
  page.on("request", (request) => origins.add(new URL(request.url()).origin));
  await page.goto("/#token=browser-test");
  await expect(page.locator("#status")).toHaveText("Connected");
  await expect(page.locator("#registry")).toContainText("unknown");
  expect(new URL(page.url()).hash).toBe("");
  await page.locator('[data-view="explorer"]').click();
  await expect(page.locator("#sources details").first()).toHaveAttribute("open", "");
  await expect(page.locator("#selection")).toContainText("signals: 1");
  await page.locator('[data-signal-id="signal"]').click();
  await expect.poll(() => page.evaluate(() => document.getElementById("plot").data?.[0]?.y?.length)).toBe(1000);
  const trace = await page.evaluate(() => document.getElementById("plot").data[0]);
  expect(trace.y.slice(0, 5)).toEqual([1, null, 3, null, 5]);
  expect(trace.connectgaps).toBe(false);
  await page.locator("#raw summary").click();
  await expect(page.locator("#raw tbody tr")).toHaveCount(200);
  await page.getByRole("button", { name: "Next values" }).click();
  await page.locator("#raw summary").click();
  await expect(page.locator("#raw tbody tr").first()).toContainText("200");
  await page.locator("#fit").click();
  await expect(page.locator("#axis-label")).toContainText("Reduced");
  expect(await page.evaluate(() => document.getElementById("plot").data[0].mode)).toBe("markers");
  expect([...origins]).toEqual(["http://127.0.0.1:8765"]);
});

test("a stale completed window cannot repopulate a reset plot", async ({ page }) => {
  await page.goto("/#token=browser-test");
  await expect(page.locator("#status")).toHaveText("Connected");
  await page.locator('[data-view="explorer"]').click();
  await expect(page.locator("#sources details").first()).toHaveAttribute("open", "");
  await page.route("**/api/v1/jobs/*/result", async (route) => {
    const response = await route.fetch();
    await new Promise((resolve) => setTimeout(resolve, 300));
    await route.fulfill({ response });
  });
  await page.locator('[data-signal-id="signal"]').click();
  await page.locator("#reset").click();
  await page.waitForTimeout(600);
  expect(await page.evaluate(() => document.getElementById("plot").data?.length || 0)).toBe(0);
  await expect(page.locator("#raw")).toBeEmpty();
});

test("missing credentials cannot query dataset facts", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#status")).toContainText("launch URL");
  const response = await page.request.post("/api/v1/records/query", { data: {} });
  expect(response.status()).toBe(401);
});

test("tensor component selection preserves a missing timestep", async ({ page }) => {
  await page.goto("/#token=browser-test");
  await expect(page.locator("#status")).toHaveText("Connected");
  await page.locator('[data-view="explorer"]').click();
  await expect(page.locator("#sources details").first()).toHaveAttribute("open", "");
  await page.locator('[data-signal-id="tensor"]').click();
  await page.getByLabel("Dimension 1").fill("1");
  await page.getByRole("button", { name: "Inspect component" }).click();
  await expect.poll(() => page.evaluate(() => document.getElementById("plot").data?.[0]?.y)).toEqual([10, null, 30]);
});

test("task relationships can continue beyond two hundred targets", async ({ page }) => {
  await page.goto("/#token=browser-test");
  await expect(page.locator("#status")).toHaveText("Connected");
  await page.locator('[data-view="tasks"]').click();
  await expect(page.locator('[data-task-id="many"]')).toHaveClass(/selected/);
  await expect(page.locator("#task-selection")).toContainText("many");
  await expect(page.locator("#task-detail .target-card")).toHaveCount(200);
  await page.getByRole("button", { name: "Next relationships" }).click();
  await expect(page.locator("#task-detail .target-card")).toHaveCount(5);
  await expect(page.locator("#task-detail .target-card").first()).toContainText("answer-200");
});

test("frontend-owned stylesheet is local and navigation renders component views", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/#token=browser-test");
  await expect(page.locator("#status")).toHaveText("Connected");
  await expect(page.locator('link[rel="stylesheet"]')).toHaveAttribute("href", "/static/app.css");
  await page.locator('[data-view="annotations"]').click();
  await expect(page.locator("#annotation-owner")).toBeVisible();
  await expect(page.locator("#annotation-list button").first()).toHaveClass(/selected/);
  await expect(page.locator("#annotation-selection")).toContainText("Selected: reviewed");
  await page.locator('[data-view="overview"]').click();
  await expect(page.locator("#overview-details")).toContainText("Values backend");
  expect(errors).toEqual([]);
});

test("record search discards an obsolete response", async ({ page }) => {
  await page.route("**/api/v1/records/query", async (route) => {
    const query = route.request().postDataJSON().query;
    if (query === "slow") await new Promise((resolve) => setTimeout(resolve, 250));
    if (!query) return route.continue();
    await route.fulfill({ json: { items: [{ record_id: query }], next_cursor: null } });
  });
  await page.goto("/#token=browser-test");
  await expect(page.locator("#status")).toHaveText("Connected");
  await page.locator('[data-view="explorer"]').click();
  await page.locator("#search").fill("slow");
  await page.locator("#search").fill("latest");
  await expect(page.locator('[data-record-id="latest"]')).toBeVisible();
  await page.waitForTimeout(350);
  await expect(page.locator('[data-record-id="slow"]')).toHaveCount(0);
});

test("leaving records cancels a job admitted after the view unmounts", async ({ page }) => {
  let admission;
  const admitted = new Promise((resolve) => { admission = resolve; });
  await page.route("**/api/v1/jobs/windows", async (route) => {
    const response = await route.fetch();
    admission(await response.json());
    await new Promise((resolve) => setTimeout(resolve, 300));
    await route.fulfill({ response });
  });
  await page.goto("/#token=browser-test");
  await expect(page.locator("#status")).toHaveText("Connected");
  await page.locator('[data-view="explorer"]').click();
  await expect(page.locator("#sources details").first()).toHaveAttribute("open", "");
  await page.locator('[data-signal-id="signal"]').click();
  const job = await admitted;
  const cancelled = page.waitForRequest((request) => request.method() === "DELETE" && request.url().endsWith(`/jobs/${job.job_id}`));
  await page.locator('[data-view="overview"]').click();
  await cancelled;
  await expect(page.locator("#plot")).toHaveCount(0);
});
