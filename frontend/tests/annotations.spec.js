import { test, expect } from "@playwright/test";

test("source annotation navigation preserves its source and temporal location", async ({
  page,
}) => {
  await page.goto("/#token=browser-test");
  await page.locator('[data-view="annotations"]').click();
  await page.locator("#annotation-owner").selectOption("Source");
  await page
    .locator("#annotation-list button")
    .filter({ hasText: "source-event" })
    .click();
  await page.getByRole("button", { name: "Inspect annotated object" }).click();
  await expect(page.locator('[data-signal-id="z-signal"]')).toHaveClass(
    "selected",
  );
  await expect(page.locator("#raw summary")).toContainText("1 displayed");
  await page.locator("#raw summary").click();
  await expect(page.locator("#raw tbody tr").first()).toContainText(
    "100000000",
  );
  await expect(page.locator("#plot-annotations")).toContainText("source-event");
});

test("point and interval annotations use the selected temporal subplot", async ({
  page,
}) => {
  await page.goto("/#token=browser-test");
  await page.locator('[data-view="explorer"]').click();
  await page.locator('[data-signal-id="ordinal"]').click();
  await page
    .locator('[data-signal-id="signal"]')
    .click({ modifiers: ["Control"] });
  await expect
    .poll(() =>
      page.evaluate(
        () => document.getElementById("plot").layout?.shapes?.length,
      ),
    )
    .toBe(3);
  const layout = await page.evaluate(
    () => document.getElementById("plot").layout,
  );
  expect(
    layout.shapes.map(({ type, xref, yref, x0, x1 }) => ({
      type,
      xref,
      yref,
      x0,
      x1,
    })),
  ).toEqual(
    expect.arrayContaining([
      { type: "line", xref: "x2", yref: "y2 domain", x0: 500000, x1: 500000 },
      { type: "rect", xref: "x2", yref: "y2 domain", x0: 250000, x1: 750000 },
      { type: "line", xref: "x2", yref: "y2 domain", x0: 125000, x1: 125000 },
    ]),
  );
  expect(
    layout.annotations.map(({ text, xref, yref, x }) => ({
      text,
      xref,
      yref,
      x,
    })),
  ).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ text: "event", xref: "x2", yref: "y2 domain" }),
      { text: "interval", xref: "x2", yref: "y2 domain", x: 500000 },
    ]),
  );
  const overlay = page.waitForResponse(
    (response) =>
      response.url().endsWith("/annotations/window") &&
      response.request().postDataJSON().signal_id === "ordinal",
  );
  await page.locator('[data-signal-id="ordinal"]').click();
  // A record-owned temporal event still reaches the UI for the ordinal signal.
  expect((await (await overlay).json()).items).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        name: "record-event",
        span_type: "point",
        start_us: "125000",
      }),
    ]),
  );
  await expect(page.locator("#plot-annotations")).toContainText("record-event");
  await expect
    .poll(() =>
      page.evaluate(() => document.getElementById("plot").data?.[0]?.y),
    )
    .toEqual([1, 2, 3]);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.getElementById("plot").layout?.shapes?.length,
      ),
    )
    .toBe(0);
});
