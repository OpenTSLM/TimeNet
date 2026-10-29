import { useEffect, useState } from "react";
import type { ViewerProps } from "./navigation";

// List items shown in a collapsed value before the reader expands it.
const LIST_PREVIEW = 3;

export type Annotation = {
  key: string; value_type: string | null; value: unknown; unit: string | null;
  description: string | null; source: string | null; confidence: number | null;
  metadata: Record<string, unknown>; occurrence_metadata: Record<string, unknown>;
};

// Optional facts get a column only when at least one row carries them.
const OPTIONAL_COLUMNS = [
  ["Unit", (a: Annotation) => a.unit],
  ["Description", (a: Annotation) => a.description],
  ["Provenance", (a: Annotation) => a.source],
  ["Confidence", (a: Annotation) => a.confidence === null ? null : String(a.confidence)],
] as const;

function metadataOf(annotation: Annotation): Record<string, unknown> | null {
  const merged = { ...annotation.metadata, ...annotation.occurrence_metadata };
  return Object.keys(merged).length ? merged : null;
}

function AnnotationValue({ value }: { value: unknown }) {
  if (value === null) return <span className="task-muted">No value</span>;
  if (!Array.isArray(value)) return <>{String(value)}</>;
  const preview = value.slice(0, LIST_PREVIEW).map(String).join(", ");
  return <details className="annotation-list">
    <summary>{value.length.toLocaleString()} {value.length === 1 ? "item" : "items"}
      {value.length > 0 && <span className="task-muted"> · {preview}{value.length > LIST_PREVIEW && ", …"}</span>}</summary>
    <ol>{value.map((item, index) => <li key={index}>{String(item)}</li>)}</ol>
  </details>;
}

export function AnnotationTable({ annotations, label }: { annotations: Annotation[]; label: string }) {
  const columns = OPTIONAL_COLUMNS.filter(([, read]) => annotations.some(a => read(a) !== null));
  const withMetadata = annotations.some(a => metadataOf(a) !== null);
  return <div className="task-table-wrap">
    <table className="annotation-table" aria-label={label}>
      <thead><tr>
        <th>Key</th><th>Value</th>
        {columns.map(([title]) => <th key={title}>{title}</th>)}
        {withMetadata && <th>Metadata</th>}
      </tr></thead>
      <tbody>{annotations.map((annotation, index) => {
        const metadata = metadataOf(annotation);
        return <tr key={index}>
          <td data-label="Key"><code>{annotation.key}</code></td>
          <td data-label="Value"><AnnotationValue value={annotation.value} /></td>
          {columns.map(([title, read]) => <td key={title} data-label={title}>{read(annotation) ?? "—"}</td>)}
          {withMetadata && <td data-label="Metadata">{metadata ? <details><summary>Show</summary>
            <pre className="task-json">{JSON.stringify(metadata, null, 2)}</pre></details> : "—"}</td>}
        </tr>;
      })}</tbody>
    </table>
  </div>;
}

export function DatasetAnnotations({ dataset, version, getJson }: Pick<ViewerProps, "dataset" | "version" | "getJson">) {
  const identity = `${dataset}@${version}`;
  const [result, setResult] = useState<{ identity: string; annotations?: Annotation[]; error?: string } | null>(null);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setResult(null);
    const query = new URLSearchParams({ dataset_id: dataset, version });
    getJson<{ annotations: Annotation[] }>(`/api/v1/dataset-annotations?${query}`, controller.signal).then(body => {
      if (!controller.signal.aborted) setResult({ identity, annotations: body.annotations });
    }).catch(reason => {
      if (!controller.signal.aborted)
        setResult({ identity, error: reason instanceof Error ? reason.message : "Annotation request failed." });
    });
    return () => controller.abort();
  }, [dataset, version, identity, retry, getJson]);
  const current = result?.identity === identity ? result : null;
  return <section className="raw annotation-section" aria-label="Dataset annotations">
    <h2>Dataset annotations{current?.annotations && ` (${current.annotations.length.toLocaleString()})`}</h2>
    <p className="annotation-note">Annotations attached to the dataset itself, not to one of its records.</p>
    {current?.error ? <div role="alert"><p>{current.error}</p><button onClick={() => setRetry(v => v + 1)}>Retry annotations</button></div>
      : !current?.annotations ? <p role="status">Loading dataset annotations…</p>
      : current.annotations.length === 0 ? <p className="empty">This dataset has no dataset-level annotations.</p>
      : <AnnotationTable annotations={current.annotations} label="Dataset annotations" />}
  </section>;
}
