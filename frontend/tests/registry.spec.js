import { test, expect } from "@playwright/test";

test("catalog search finds datasets beyond the first page", async ({
  page,
}) => {
  await page.request.post("http://127.0.0.1:8766/api/v1/registry/close", {
    headers: { Authorization: "Bearer browser-test" },
    data: {},
  });
  await page.goto("http://127.0.0.1:8766/#token=browser-test");
  await expect(page.locator(".catalog-card")).toHaveCount(50);
  await expect(
    page.getByRole("button", { name: "Open Hidden needle", exact: true }),
  ).toHaveCount(0);
  await page.getByLabel("Search", { exact: true }).fill("hidden needle");
  await expect(
    page.getByRole("button", { name: "Open Hidden needle", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".catalog-card")).toHaveCount(1);
  await page.getByLabel("Search", { exact: true }).fill("");
  await expect(page.locator(".catalog-card")).toHaveCount(50);
});

test.describe("real registry switching", () => {
  test.beforeEach(async ({ page }) => {
    await page.request.post("http://127.0.0.1:8766/api/v1/registry/close", {
      headers: { Authorization: "Bearer browser-test" },
      data: {},
    });
    await page.goto("http://127.0.0.1:8766/#token=browser-test");
    await page
      .getByRole("button", { name: "Open Dataset A", exact: true })
      .click();
    await page.locator('[data-view="tasks"]').click();
    await expect(page.locator("#task-selection")).toContainText("Task: task-a");
  });

  test("switching datasets clears task and record navigation", async ({
    page,
  }) => {
    await page.getByRole("button", { name: "Inspect target" }).click();
    await expect(page.locator("#selection")).toContainText("Record: record-a;");
    await expect(
      page.getByRole("button", { name: "Return to task" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Datasets", exact: true }).click();
    await page
      .getByRole("button", { name: "Open Dataset B", exact: true })
      .click();
    await expect(page.locator("#overview-details")).toContainText(
      "test/b@1.0.0",
    );
    await page.locator('[data-view="explorer"]').click();
    await expect(page.locator("#selection")).toContainText("Record: record-b;");
    await expect(page.locator('[data-record-id="record-a"]')).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: "Return to task" }),
    ).toHaveCount(0);
    await expect
      .poll(() =>
        page.evaluate(() => document.getElementById("plot").data?.[0]?.y),
      )
      .toEqual([22, 23]);
    await page.locator('[data-view="tasks"]').click();
    await expect(page.locator("#task-selection")).toContainText("Task: task-b");
    await expect(page.locator('[data-task-id="task-a"]')).toHaveCount(0);
    await page.getByRole("button", { name: "Inspect target" }).click();
    await expect(page.locator("#selection")).toContainText("Record: record-b;");
    await expect(page.locator('[role="alert"]')).toHaveCount(0);
  });

  test("a pending owner lookup cannot navigate back into the previous dataset", async ({
    page,
  }) => {
    let release, captured;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    const ready = new Promise((resolve) => {
      captured = resolve;
    });
    await page.route("**/api/v1/owners/record", async (route) => {
      // Keep the real response and delay only its delivery to reproduce the race.
      const response = await route.fetch();
      expect(response.status()).toBe(200);
      expect((await response.json()).record_id).toBe("record-a");
      captured();
      await gate;
      await route.fulfill({ response });
    });
    try {
      await page.getByRole("button", { name: "Inspect target" }).click();
      await ready;
      await page.getByRole("button", { name: "Datasets", exact: true }).click();
      await page
        .getByRole("button", { name: "Open Dataset B", exact: true })
        .click();
      await expect(page.locator("#overview-details")).toContainText(
        "test/b@1.0.0",
      );
      const delivered = page.waitForResponse("**/api/v1/owners/record");
      release();
      await (await delivered).finished();
      await page.evaluate(
        () =>
          new Promise((resolve) =>
            requestAnimationFrame(() => requestAnimationFrame(resolve)),
          ),
      );
      await expect(page.locator("#overview-details")).toContainText(
        "test/b@1.0.0",
      );
      await expect(page.locator("#records-workspace")).toHaveCount(0);
    } finally {
      release();
    }
  });
});
