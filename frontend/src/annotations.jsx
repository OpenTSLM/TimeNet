import { useEffect, useState } from "react";
import { useResource, useCursor } from "./hooks";
import { Pagination } from "./components";

export function annotationLabel(annotation) {
  const span = annotation.span_type === "static" ? "" : ` (${annotation.span_type}: ${annotation.start_us ?? "…"}–${annotation.end_us ?? "…"})`;
  return `${annotation.name}${annotation.value === null ? "" : ` = ${annotation.value}`}${span} → ${annotation.object_type}: ${annotation.object_id}`;
}

export function Annotations({ client, openAnnotation }) {
  const [filters, setFilters] = useState({ object_type: "", object_id: "", span_type: "", value_query: "" });
  const [selected, setSelected] = useState(null);
  const paging = useCursor(JSON.stringify(filters));
  const body = Object.fromEntries(Object.entries(filters).map(([key, value]) => [key, value || null]));
  const page = useResource(client, "/api/v1/annotations/query", { ...body, after: paging.after, limit: 10 });
  useEffect(() => {
    if (!selected && page.data?.items.length) setSelected(page.data.items[0]);
  }, [page.data, selected]);
  const change = (key) => (event) => setFilters({ ...filters, [key]: event.target.value });
  return <><section className="panel"><h2>Annotations</h2><div className="filters">
    <select id="annotation-owner" value={filters.object_type} onChange={change("object_type")}><option value="">All owner types</option>
      {["Dataset", "Record", "Source", "Signal", "Task"].map((name) => <option key={name}>{name}</option>)}</select>
    <select id="annotation-span" value={filters.span_type} onChange={change("span_type")}><option value="">All span types</option>
      {["static", "point", "interval"].map((name) => <option key={name}>{name}</option>)}</select>
    <input id="annotation-owner-id" placeholder="Owner ID" value={filters.object_id} onChange={change("object_id")} />
    <input id="annotation-value" placeholder="Search value" value={filters.value_query} onChange={change("value_query")} />
  </div><div id="annotation-list">{page.data?.items.map((item) => <button key={JSON.stringify([item.dataset_id, item.version, item.occurrence_id])}
    data-occurrence-id={item.occurrence_id} className={selected === item ? "selected" : ""} onClick={() => setSelected(item)}>{annotationLabel(item)}</button>)}
    <Pagination paging={paging} next={page.data?.next_cursor} /></div>{page.error && <p role="alert">{page.error}</p>}
  </section><section className="panel"><h2>Annotation context</h2><div id="annotation-selection" className="muted">{selected ? `Selected: ${selected.name} (${selected.occurrence_id})` : "Select an annotation."}</div>
    <div id="annotation-detail">{selected && <><h3>{selected.name}</h3><button onClick={() => openAnnotation(selected)}>Inspect annotated object</button>
      <p>Occurrence ID: {selected.occurrence_id}</p><p>Content ID: {selected.content_id}</p><p>Owner: {selected.object_type}: {selected.object_id}</p>
      <p>Value: {selected.value ?? "(none)"}</p><p>Span: {selected.span_type === "static" ? "static" : `${selected.start_us ?? "…"} – ${selected.end_us ?? "…"}`}</p></>}</div>
  </section></>;
}
