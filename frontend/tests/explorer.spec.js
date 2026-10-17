import { test, expect } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/#token=browser-test");
  await page.locator('[data-view="explorer"]').click();
  await page.locator('[data-signal-id="signal"]').click();
});

test("plots preserve gaps and raw values continue to the next page", async ({
  page,
}) => {
  await expect
    .poll(() =>
      page.evaluate(() =>
        document.getElementById("plot").data?.[0]?.y?.slice(0, 5),
      ),
    )
    .toEqual([1, null, 3, null, 5]);
  expect(
    await page.evaluate(
      () => document.getElementById("plot").data[0].connectgaps,
    ),
  ).toBe(false);
  await page.locator("#raw summary").click();
  await expect(page.locator("#raw tbody tr")).toHaveCount(200);
  await page.getByRole("button", { name: "Next values" }).click();
  await page.locator("#raw summary").click();
  await expect(page.locator("#raw tbody tr").first()).toContainText("200");
});

test("oversized plots recover when the user selects a bounded window", async ({
  page,
}) => {
  await page.locator("#fit").click();
  await expect(
    page.getByRole("alert").filter({ hasText: "20,000 steps" }).first(),
  ).toBeVisible();
  await page.locator("#stop").fill("20000");
  await page.locator("#window").click();
  await expect(page.locator("#axis-label")).toContainText("Reduced");
  await expect
    .poll(() =>
      page.evaluate(() => document.getElementById("plot").data?.[0]?.mode),
    )
    .toBe("markers");
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("tensor components preserve missing timesteps", async ({ page }) => {
  await page.locator('[data-signal-id="tensor"]').click();
  await page.getByLabel("Dimension 1").fill("1");
  await page.getByRole("button", { name: "Inspect component" }).click();
  await expect
    .poll(() =>
      page.evaluate(() => document.getElementById("plot").data?.[0]?.y),
    )
    .toEqual([10, null, 30]);
});

test("reset ignores an in-flight plot response", async ({ page }) => {
  let release, captured, delivered;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const ready = new Promise((resolve) => {
    captured = resolve;
  });
  const finished = new Promise((resolve) => {
    delivered = resolve;
  });
  await page.route("**/api/v1/windows", async (route) => {
    if (route.request().postDataJSON().mode !== "plot") return route.continue();
    const response = await route.fetch();
    captured();
    await gate;
    await route.fulfill({ response });
    delivered();
  });
  try {
    await page.locator("#stop").fill("100");
    await page.locator("#window").click();
    await ready;
    await page.locator("#reset").click();
    release();
    await finished;
    await page.evaluate(
      () =>
        new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)),
        ),
    );
    expect(
      await page.evaluate(
        () => document.getElementById("plot").data?.length || 0,
      ),
    ).toBe(0);
    await expect(page.locator("#raw")).toBeEmpty();
  } finally {
    release();
    await page.unrouteAll({ behavior: "wait" });
  }
});
