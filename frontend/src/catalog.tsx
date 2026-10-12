import type { CatalogItem } from "./types";
import type { Schemas } from "./api";

export function Catalog({
  datasets,
  registry,
  openDataset,
  issues,
  next,
  onNext,
  onFirst,
  filters,
  setFilters,
  facets,
  total,
}: {
  datasets: CatalogItem[];
  registry?: string;
  openDataset: (item: CatalogItem) => void;
  issues: Schemas["CatalogIssue"][];
  next?: string | null;
  onNext: (cursor: string) => void;
  onFirst: () => void;
  filters: Schemas["CatalogQuery"];
  setFilters: (filters: Schemas["CatalogQuery"]) => void;
  facets: Schemas["CatalogFacets"];
  total: number;
}) {
  const { query = "", domain = "", task = "", license = "" } = filters;
  const options = facets;
  const visible = datasets;
  return (
    <main id="catalog" className="catalog">
      <section className="panel">
        <h2>Available datasets</h2>
        <p className="muted">{registry}</p>
        <div className="filters">
          <label>
            Search
            <input
              value={query}
              onChange={(event) =>
                setFilters({ ...filters, query: event.target.value })
              }
            />
          </label>
          <label>
            Domain
            <select
              value={domain}
              onChange={(event) =>
                setFilters({ ...filters, domain: event.target.value })
              }
            >
              <option value="">All domains</option>
              {options.domains.map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label>
            Task
            <select
              value={task}
              onChange={(event) =>
                setFilters({ ...filters, task: event.target.value })
              }
            >
              <option value="">All tasks</option>
              {options.tasks.map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label>
            License
            <select
              value={license}
              onChange={(event) =>
                setFilters({ ...filters, license: event.target.value })
              }
            >
              <option value="">All licenses</option>
              {options.licenses.map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
        </div>
        <p className="muted">
          {total} matching dataset{total === 1 ? "" : "s"}; {visible.length} on
          this page
        </p>
        {issues.map((issue) => (
          <p role="alert" key={issue.dataset_id}>
            {issue.dataset_id}: {issue.message}
          </p>
        ))}
        <button onClick={onFirst}>First datasets</button>
        {next && <button onClick={() => onNext(next)}>Next datasets</button>}
        <div className="catalog-list">
          {visible.map((item) => (
            <article
              className="catalog-card"
              key={item.dataset_id}
              role="button"
              tabIndex={0}
              aria-label={`Open ${item.name}`}
              onClick={() => openDataset(item)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  openDataset(item);
                }
              }}
            >
              <h3>{item.name}</h3>
              <code>
                {item.dataset_id}@{item.version}
              </code>
              <p>
                {item.description}
                {item.description_truncated ? "… (truncated preview)" : ""}
              </p>
              <p>
                <strong>Domains:</strong> {item.domains.join(", ") || "—"}
                <br />
                <strong>Tasks:</strong> {item.tasks.join(", ") || "—"}
                <br />
                <strong>License:</strong> {item.license}
              </p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
