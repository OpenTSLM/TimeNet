// Custom behavior for the TimeNet docs: the "Copy page" dropdown and the datasets catalog.
// Loaded after the Zensical theme bundle, so window.document$ is available and re-emits on
// instant navigation; both initializers are idempotent via a data-ready guard.

function initCopyPage() {
  document.querySelectorAll(".copy-page:not([data-ready])").forEach((root) => {
    root.setAttribute("data-ready", "");
    const toggle = root.querySelector(".copy-page__toggle");
    const menu = root.querySelector(".copy-page__menu");
    const status = root.querySelector('[role="status"]');
    const items = () => Array.from(menu.querySelectorAll('[role="menuitem"]'));
    const isOpen = () => toggle.getAttribute("aria-expanded") === "true";

    const setOpen = (open) => {
      toggle.setAttribute("aria-expanded", String(open));
      menu.hidden = !open;
      if (open) {
        const first = items()[0];
        if (first) first.focus();
      }
    };

    toggle.addEventListener("click", (e) => {
      e.stopPropagation();
      setOpen(!isOpen());
    });

    toggle.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        setOpen(true);
      }
    });

    root.querySelectorAll("[data-copy]").forEach((btn) => {
      const label = btn.querySelector(".copy-page__title") || btn;
      const original = label.textContent; // captured once, so a rapid re-click can't stick it
      let resetTimer;
      btn.addEventListener("click", async () => {
        // Directory URLs (the site root "/TimeNet/") have no ".html" — target index.md there.
        const path = location.pathname;
        const url = path.endsWith("/") ? path + "index.md" : path.replace(/\.html$/, ".md");
        try {
          const md = await (await fetch(url)).text();
          await navigator.clipboard.writeText(md);
          label.textContent = "Copied!";
          if (status) status.textContent = "Page copied as Markdown";
        } catch (err) {
          label.textContent = "Copy failed";
          if (status) status.textContent = "Copy failed";
        }
        clearTimeout(resetTimer);
        resetTimer = setTimeout(() => {
          label.textContent = original;
        }, 2000);
        if (isOpen()) setOpen(false);
      });
    });

    menu.addEventListener("keydown", (e) => {
      const list = items();
      const i = list.indexOf(document.activeElement);
      if (e.key === "ArrowDown") {
        e.preventDefault();
        (list[i + 1] || list[0]).focus();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        (list[i - 1] || list[list.length - 1]).focus();
      } else if (e.key === "Home") {
        e.preventDefault();
        list[0].focus();
      } else if (e.key === "End") {
        e.preventDefault();
        list[list.length - 1].focus();
      } else if (e.key === "Escape") {
        e.preventDefault();
        setOpen(false);
        toggle.focus();
      }
    });

    document.addEventListener("click", (e) => {
      if (isOpen() && !root.contains(e.target)) setOpen(false);
    });
  });
}

function initCatalog() {
  const root = document.querySelector(".ds-catalog:not([data-ready])");
  if (!root) return;
  root.setAttribute("data-ready", "");

  const search = root.querySelector(".ds-search");
  const countEl = root.querySelector(".ds-count");
  const clearBtn = root.querySelector(".ds-clear");
  const rows = Array.from(root.querySelectorAll(".ds-row"));
  const dropdowns = Array.from(root.querySelectorAll(".ds-dd"));
  const facetAttr = { domain: "data-domains", tag: "data-tags" };

  const rowValues = (row, facet) => (row.getAttribute(facetAttr[facet]) || "").split(/\s+/).filter(Boolean);
  const selectedIn = (dd) => Array.from(dd.querySelectorAll("input:checked")).map((i) => i.value);
  const detailOf = (row) => row.nextElementSibling; // the paired .ds-detail-row
  const isOpen = (row) => row.getAttribute("aria-expanded") === "true";

  const apply = () => {
    const terms = (search.value || "").toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    rows.forEach((row) => {
      const text = (row.getAttribute("data-text") || "").toLowerCase();
      const okText = terms.every((t) => text.includes(t));
      const okFacets = dropdowns.every((dd) => {
        const sel = selectedIn(dd);
        if (!sel.length) return true;
        const vals = rowValues(row, dd.dataset.facet);
        return sel.some((v) => vals.includes(v));
      });
      const show = okText && okFacets;
      row.hidden = !show;
      const detail = detailOf(row);
      if (detail) detail.hidden = !show || !isOpen(row);
      if (show) shown += 1;
    });
    if (countEl) countEl.textContent = `${shown} of ${rows.length} datasets`;
    dropdowns.forEach((dd) => {
      const badge = dd.querySelector(".ds-dd__count");
      const n = selectedIn(dd).length;
      if (badge) badge.textContent = n ? ` (${n})` : "";
    });
  };

  const toggleRow = (row) => {
    const open = isOpen(row);
    row.setAttribute("aria-expanded", String(!open));
    const detail = detailOf(row);
    if (detail) detail.hidden = open || row.hidden;
  };

  rows.forEach((row) => {
    row.addEventListener("click", () => toggleRow(row));
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleRow(row);
      }
    });
  });

  const closeAll = (except) => {
    dropdowns.forEach((dd) => {
      if (dd === except) return;
      dd.querySelector(".ds-dd__menu").hidden = true;
      dd.querySelector(".ds-dd__btn").setAttribute("aria-expanded", "false");
    });
  };

  dropdowns.forEach((dd) => {
    const btn = dd.querySelector(".ds-dd__btn");
    const menu = dd.querySelector(".ds-dd__menu");
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = btn.getAttribute("aria-expanded") === "true";
      closeAll(dd);
      btn.setAttribute("aria-expanded", String(!open));
      menu.hidden = open;
    });
    menu.addEventListener("click", (e) => e.stopPropagation());
    menu.addEventListener("change", apply);
  });

  document.addEventListener("click", () => closeAll(null));

  search.addEventListener("input", apply);
  clearBtn.addEventListener("click", () => {
    search.value = "";
    root.querySelectorAll(".ds-dd input:checked").forEach((i) => {
      i.checked = false;
    });
    apply();
  });

  apply();
}

if (window.document$) {
  window.document$.subscribe(() => {
    initCopyPage();
    initCatalog();
  });
} else {
  document.addEventListener("DOMContentLoaded", () => {
    initCopyPage();
    initCatalog();
  });
}
