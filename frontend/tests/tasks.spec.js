import { test, expect } from "@playwright/test";

test("task relationships can continue beyond two hundred targets", async ({
  page,
}) => {
  await page.goto("/#token=browser-test");
  await expect(page.locator("#status")).toHaveText("Connected");
  await page.locator('[data-view="tasks"]').click();
  await expect(page.locator('[data-task-id="many"]')).toHaveClass(/selected/);
  await expect(page.locator("#task-selection")).toContainText("many");
  await expect(page.locator("#task-detail .target-card")).toHaveCount(200);
  await page.getByRole("button", { name: "Next relationships" }).click();
  await expect(page.locator("#task-detail .target-card")).toHaveCount(5);
  await expect(page.locator("#task-detail .target-card").first()).toContainText(
    "answer-200",
  );
});
