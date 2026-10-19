import { useEffect, useState, type FormEvent } from "react";
import { TasksBrowser } from "./tasks";
import { RecordsBrowser } from "./records";

type Metadata = {
  dataset_id: string;
  dataset_version: string;
  name: string;
  description: string;
  license: string;
  access: string;
  domains: string[];
  tags: string[];
  source_url: string | null;
  citation: string | null;
};

type Counts = {
  records: number;
  sources: number;
  signals: number;
  annotation_occurrences: number;
  tasks: Record<string, number>;
};

type Manifest = {
  metadata: Metadata;
  counts: Counts;
  schema: {
    time_series_specs: { spec_type: string }[];
    annotations: { key: string }[];
    tasks: { task_type: string }[];
  };
};

type CatalogItem = {
  metadata: Metadata;
  counts: Counts;
  task_types: string[];
  signal_types: string[];
  annotation_types: string[];
};

type CatalogPage = {
  items: CatalogItem[];
  total: number;
  offset: number;
  limit: number;
  domains: string[];
  licenses: string[];
};

type Selection = { dataset_id: string; version: string };
type Filters = { query: string; domain: string; license: string };

const launchToken = new URLSearchParams(location.hash.slice(1)).get("token");
if (launchToken) sessionStorage.setItem("timenet-viewer-token", launchToken);
const token = launchToken || sessionStorage.getItem("timenet-viewer-token");
const directLaunch = new URLSearchParams(location.search).get("from") !== "catalog"
  && selectionFromUrl() !== null;
history.replaceState(null, "", location.pathname + location.search);

function selectionFromUrl(): Selection | null {
  const params = new URLSearchParams(location.search);
  const dataset_id = params.get("dataset");
  const version = params.get("version");
  return dataset_id && version ? { dataset_id, version } : null;
}

async function getJson<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    headers: { Authorization: `Bearer ${token}` },
    signal,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as {
      detail?: string | { message: string } | null; message?: string;
    } | null;
    throw new Error((typeof body?.detail === "string" ? body.detail : body?.detail?.message)
      || body?.message || `Viewer request failed (${response.status}).`);
  }
  return (await response.json()) as T;
}

function safeUrl(value: string | null): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:"
      ? url.toString()
      : null;
  } catch {
    return null;
  }
}

function taskTotal(counts: Counts): number {
  return Object.values(counts.tasks).reduce((sum, count) => sum + count, 0);
}

export function App() {
  const [search, setSearch] = useState(location.search);
  const [selection, setSelection] = useState<Selection | null>(selectionFromUrl);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [catalog, setCatalog] = useState<CatalogPage | null>(null);
  // TODO: Persist catalog filters and page in the URL if reloadable or shareable
  // catalog state becomes part of the viewer's navigation contract.
  const [filters, setFilters] = useState<Filters>({ query: "", domain: "", license: "" });
  const [applied, setApplied] = useState<Filters>(filters);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onBack = () => { setSelection(selectionFromUrl()); setSearch(location.search); };
    addEventListener("popstate", onBack);
    return () => removeEventListener("popstate", onBack);
  }, []);

  useEffect(() => {
    if (!token) {
      setError("Open the authenticated URL printed by timenet view.");
      return;
    }
    const controller = new AbortController();
    setError(null);
    if (selection) {
      setManifest(null);
      const params = new URLSearchParams({
        dataset_id: selection.dataset_id,
        version: selection.version,
      });
      getJson<Manifest>(`/api/v1/manifest?${params}`, controller.signal)
        .then(setManifest)
        .catch((reason: unknown) => {
          if (!controller.signal.aborted)
            setError(reason instanceof Error ? reason.message : "Viewer request failed.");
        });
    } else {
      setCatalog(null);
      const params = new URLSearchParams({
        ...applied,
        offset: String(offset),
        limit: "24",
      });
      getJson<CatalogPage>(`/api/v1/catalog?${params}`, controller.signal)
        .then(setCatalog)
        .catch((reason: unknown) => {
          if (!controller.signal.aborted)
            setError(reason instanceof Error ? reason.message : "Viewer request failed.");
        });
    }
    return () => controller.abort();
  }, [selection, applied, offset]);

  function openDataset(item: CatalogItem) {
    const next = {
      dataset_id: item.metadata.dataset_id,
      version: item.metadata.dataset_version,
    };
    history.pushState(null, "", `/?${new URLSearchParams({ dataset: next.dataset_id, version: next.version, from: "catalog" })}`);
    setSearch(location.search);
    setSelection(next);
  }

  function openCatalog() {
    history.pushState(null, "", "/");
    setSearch(location.search);
    setSelection(null);
  }

  function submitFilters(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setOffset(0);
    setApplied(filters);
  }

  function navigateDataset(params: URLSearchParams, options?: { replace?: boolean }) {
    if (options?.replace) history.replaceState(null, "", `/?${params}`);
    else history.pushState(null, "", `/?${params}`);
    setSearch(location.search);
  }

  if (selection && ["records", "source"].includes(new URLSearchParams(search).get("view") || "")) {
    return <RecordsBrowser key={`${selection.dataset_id}@${selection.version}`} dataset={selection.dataset_id} version={selection.version}
      search={search} navigate={navigateDataset} getJson={getJson} />;
  }

  if (selection && new URLSearchParams(search).get("view") === "tasks") {
    return <TasksBrowser key={`${selection.dataset_id}@${selection.version}`} dataset={selection.dataset_id} version={selection.version}
      search={search} navigate={navigateDataset} getJson={getJson} />;
  }

  if (error) return (
    <main className="notice" role="alert">
      {selection && !directLaunch && <button className="back" type="button" onClick={openCatalog}>← Registry catalog</button>}
      <p>{error}</p>
    </main>
  );
  if (selection) {
    if (!manifest) return <main className="notice">Loading dataset…</main>;
    const { metadata, counts, schema } = manifest;
    const sourceUrl = safeUrl(metadata.source_url);
    return (
      <main className="layout">
        {!directLaunch && <button className="back" type="button" onClick={openCatalog}>← Registry catalog</button>}
        <header>
          <p className="eyebrow">TimeNet viewer · Dataset overview</p>
          <h1>{metadata.name}</h1>
          <p className="identity">{metadata.dataset_id} <span>·</span> {metadata.dataset_version}</p>
          <p className="description">{metadata.description}</p>
          {sourceUrl && <a href={sourceUrl} target="_blank" rel="noreferrer">Original source ↗</a>}
        </header>

        <section className="raw">
          <h2>Tasks</h2>
          <button className="task-link" onClick={() => {
            const params = new URLSearchParams(search);
            params.set("view", "tasks");
            navigateDataset(params);
          }}>Browse tasks →</button>
        </section>

        <section className="metrics" aria-label="Dataset counts">
          {[
            ["Records", counts.records],
            ["Sources", counts.sources],
            ["Signals", counts.signals],
            ["Tasks", taskTotal(counts)],
            ["Annotations", counts.annotation_occurrences],
          ].map(([label, value]) => (
            <article className="metric" key={label}>
              <span>{label}</span><strong>{Number(value).toLocaleString()}</strong>
            </article>
          ))}
        </section>

        <section className="facts" aria-label="Dataset facts">
          <div><h2>Domains</h2><p>{metadata.domains.join(", ") || "None declared"}</p></div>
          <div><h2>License</h2><p>{metadata.license}</p></div>
          <div><h2>Access</h2><p>{metadata.access}</p></div>
          <div><h2>Task types</h2><p>{schema.tasks.map((task) => task.task_type).join(", ") || "None declared"}</p></div>
          <div><h2>Signal types</h2><p>{schema.time_series_specs.map((spec) => spec.spec_type).join(", ") || "None declared"}</p></div>
          <div><h2>Annotation types</h2><p>{schema.annotations.map((item) => item.key).join(", ") || "None declared"}</p></div>
          <div><h2>Tags</h2><p>{metadata.tags.join(", ") || "None"}</p></div>
          <div><h2>Citation</h2><p>{metadata.citation || "None declared"}</p></div>
        </section>

        <details className="raw">
          <summary>Full manifest JSON</summary>
          <pre>{JSON.stringify(manifest, null, 2)}</pre>
        </details>
      </main>
    );
  }

  if (!catalog) return <main className="notice">Loading registry…</main>;
  const page = Math.floor(catalog.offset / catalog.limit) + 1;
  const pageCount = Math.max(1, Math.ceil(catalog.total / catalog.limit));
  return (
    <main className="layout">
      <header>
        <p className="eyebrow">TimeNet viewer · Registry catalog</p>
        <h1>Explore datasets</h1>
        <p className="description">Browse committed TimeF datasets by their manifest facts.</p>
      </header>

      <form className="filters" onSubmit={submitFilters}>
        <label>Search
          <input value={filters.query} onChange={(event) => setFilters({ ...filters, query: event.target.value })}
            placeholder="Name, ID, description, or tag" />
        </label>
        <label>Domain
          <select value={filters.domain} onChange={(event) => setFilters({ ...filters, domain: event.target.value })}>
            <option value="">All domains</option>
            {catalog.domains.map((domain) => <option key={domain} value={domain}>{domain}</option>)}
          </select>
        </label>
        <label>License
          <select value={filters.license} onChange={(event) => setFilters({ ...filters, license: event.target.value })}>
            <option value="">All licenses</option>
            {catalog.licenses.map((license) => <option key={license} value={license}>{license}</option>)}
          </select>
        </label>
        <button type="submit">Apply filters</button>
      </form>

      <p className="result-count">{catalog.total.toLocaleString()} datasets</p>
      {catalog.items.length === 0 && <p className="empty">No datasets match these filters.</p>}
      <section className="catalog" aria-label="Datasets">
        {catalog.items.map((item) => (
          <article className="dataset-card" key={`${item.metadata.dataset_id}@${item.metadata.dataset_version}`}>
            <div className="card-head">
              <div>
                <h2>{item.metadata.name}</h2>
                <p className="identity">{item.metadata.dataset_id} <span>·</span> {item.metadata.dataset_version}</p>
              </div>
              <button type="button" onClick={() => openDataset(item)}>Open dataset →</button>
            </div>
            <p>{item.metadata.description}</p>
            <p className="card-tags">{[...item.metadata.domains, item.metadata.license, item.metadata.access].join(" · ")}</p>
            {item.metadata.tags.length > 0 && <p className="card-types">Tags: {item.metadata.tags.join(", ")}</p>}
            {safeUrl(item.metadata.source_url) && (
              <a href={safeUrl(item.metadata.source_url)!} target="_blank" rel="noreferrer">Original source ↗</a>
            )}
            <div className="card-stats">
              <span>{item.counts.records.toLocaleString()} records</span>
              <span>{taskTotal(item.counts).toLocaleString()} tasks</span>
              <span>{item.counts.annotation_occurrences.toLocaleString()} annotations</span>
            </div>
            <p className="card-types">Task types: {item.task_types.join(", ") || "None declared"}</p>
            <p className="card-types">Signal types: {item.signal_types.join(", ") || "None declared"}</p>
            <p className="card-types">Annotation types: {item.annotation_types.join(", ") || "None declared"}</p>
          </article>
        ))}
      </section>
      {catalog.total > catalog.limit && (
        <nav className="pagination" aria-label="Catalog pages">
          <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - catalog.limit))}>Previous</button>
          <span>Page {page} of {pageCount}</span>
          <button type="button" disabled={offset + catalog.limit >= catalog.total} onClick={() => setOffset(offset + catalog.limit)}>Next</button>
        </nav>
      )}
    </main>
  );
}
