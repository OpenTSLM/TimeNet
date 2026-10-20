import { useEffect, useState } from "react";
import { TaskRecordWorkspace } from "./records";
import { changeSelection, selectRecord, selectTask } from "./navigation";
import type { ViewerProps } from "./navigation";

const TASK_PAGE_SIZE = 20;

type TaskSummary = {
  id: string;
  task_type: string;
  split: string | null;
  prompt_preview: string | null;
};
type TaskPage = { items: TaskSummary[]; offset: number; limit: number; has_more: boolean };
type Span = {
  kind: "point" | "interval";
  unit: "microseconds" | "steps";
  start: number;
  stop: number | null;
  signal_ids: string[] | null;
};
type Target =
  | { kind: "str" | "int" | "float" | "bool"; value: string | number | boolean }
  | { kind: "record" | "signal"; id: string }
  | { kind: "span"; value: Span };
type Annotation = {
  id: string;
  occurrence_id: string | null;
  key: string;
  value_type: string | null;
  value: unknown;
  span: Span | null;
  unit: string | null;
  description: string | null;
  source: string | null;
  confidence: number | null;
  metadata: Record<string, unknown>;
  occurrence_metadata: Record<string, unknown>;
};
type TaskDetail = {
  id: string;
  task_type: string;
  split: string | null;
  prompt: string | null;
  rationale: string | null;
  input_record_ids: string[];
  parent_task_ids: string[];
  input_modalities: string[];
  scope: Span | null;
  targets: Target[] | null;
  configuration: Record<string, unknown>;
  metadata: Record<string, unknown>;
  annotations: Annotation[];
  input_annotations: Annotation[];
  target_annotations: Annotation[];
};

function label(value: string): string {
  return value.replace(/_/g, " ");
}

function Json({ value }: { value: unknown }) {
  return <pre className="task-json">{JSON.stringify(value, null, 2)}</pre>;
}

export function TasksBrowser(props: ViewerProps) {
  const { dataset, version, search, navigate, getJson } = props;
  const params = new URLSearchParams(search);
  const split = params.get("split") || "";
  const rawOffset = Number(params.get("offset") || 0);
  const offset = Number.isSafeInteger(rawOffset) && rawOffset >= 0 ? rawOffset : 0;
  const taskId = params.get("task");
  const [page, setPage] = useState<TaskPage | null>(null);
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setPage(null);
    setDetail(null);
    setError(null);
    const query = new URLSearchParams({ dataset_id: dataset, version });
    if (taskId) query.set("task_id", taskId);
    else {
      query.set("offset", String(offset));
      query.set("limit", String(TASK_PAGE_SIZE));
      if (split) query.set("split", split);
    }
    const request = taskId
      ? getJson<TaskDetail>(`/api/v1/task?${query}`, controller.signal).then(value => {
        if (!controller.signal.aborted) setDetail(value);
      })
      : getJson<TaskPage>(`/api/v1/tasks?${query}`, controller.signal).then(value => {
        if (!controller.signal.aborted) setPage(value);
      });
    request.catch((reason: unknown) => {
      if (!controller.signal.aborted)
        setError(reason instanceof Error ? reason.message : "Task request failed.");
    });
    return () => controller.abort();
  }, [dataset, version, split, offset, taskId, retry, getJson]);

  function go(changes: Record<string, string | null>) {
    navigate(changeSelection(search, changes));
  }

  function openRecord(id: string) {
    navigate(selectRecord(search, id));
  }

  return <main className="layout task-layout">
    <button className="back" onClick={() => taskId
      ? navigate(selectTask(search, null))
      : go({ view: null, task: null, split: null, offset: null })}>
      ← {taskId ? "Task list" : "Dataset overview"}
    </button>
    <header>
      <p className="eyebrow">TimeNet viewer · {taskId ? "Task details" : "Tasks"}</p>
      <h1>{taskId || "Browse tasks"}</h1>
      <p className="identity">{dataset} · {version}</p>
    </header>
    {!taskId && <>
      <div className="task-toolbar">
        <div className="filters task-filters">
          <label>Split
            <select value={split} onChange={event => go({ split: event.target.value, offset: "0" })}>
              <option value="">All splits</option>
              <option value="train">Train</option>
              <option value="validation">Validation</option>
              <option value="test">Test</option>
            </select>
          </label>
        </div>
        <p className="task-range">{page && page.items.length > 0
          ? `Showing ${offset + 1}–${offset + page.items.length}`
          : `${TASK_PAGE_SIZE} tasks per page`}</p>
        <nav className="pagination task-pagination" aria-label="Task pages">
          <button disabled={offset === 0} onClick={() => go({ offset: String(Math.max(0, offset - TASK_PAGE_SIZE)) })}>Previous</button>
          <span>Page {Math.floor(offset / TASK_PAGE_SIZE) + 1}</span>
          <button disabled={!page?.has_more} onClick={() => go({ offset: String(offset + TASK_PAGE_SIZE) })}>Next</button>
        </nav>
      </div>
    </>}
    {error ? <div role="alert" className="empty"><p>{error}</p>
      <button onClick={() => setRetry(value => value + 1)}>Retry</button>
    </div> : (!page && !detail) ? <p role="status">Preparing task data… Remote versions may take time to download.</p> : null}
    {page && (page.items.length === 0
      ? <p className="empty">No tasks on this page for the selected split.</p>
      : <div className="task-table-wrap"><table className="task-table">
        <thead><tr><th scope="col">Task ID</th><th scope="col">Task type</th><th scope="col">Split</th><th scope="col">Prompt preview</th></tr></thead>
        <tbody>{page.items.map(item => <tr key={item.id}>
          <td data-label="Task ID"><button className="task-link task-id" onClick={() => navigate(selectTask(search, item.id))}>
            {item.id}<span aria-hidden="true" className="task-open">↗</span>
          </button></td>
          <td data-label="Task type"><span className="task-type">{label(item.task_type)}</span></td>
          <td data-label="Split"><span className={`task-split task-split-${item.split || "none"}`}>{item.split || "Unassigned"}</span></td>
          <td data-label="Prompt preview"><p className={`task-preview${item.prompt_preview === null ? " task-muted" : ""}`}
            title={item.prompt_preview ?? undefined}>{item.prompt_preview ?? "No prompt"}</p></td>
        </tr>)}</tbody>
      </table></div>)}
    {detail?.id === taskId && <>
      <section className="raw task-prompt"><h2>Prompt</h2>
        <p>{detail.prompt ?? "No prompt"}</p>
      </section>
      {detail.rationale && <section className="raw task-prompt"><h2>Rationale</h2>
        <p>{detail.rationale}</p>
      </section>}
      <section className="facts task-facts" aria-label="Task facts">
        <div><h2>Task type</h2><p className="task-type">{label(detail.task_type)}</p></div>
        <div><h2>Split</h2><p>{detail.split || "Unassigned"}</p></div>
        <div><h2>Input record IDs</h2><p>{detail.input_record_ids.length ? detail.input_record_ids.map(id =>
          <button className="task-link record-task" key={id} onClick={() => openRecord(id)}>{id}</button>) : "None"}</p></div>
        <div><h2>Parent task IDs</h2><p>{detail.parent_task_ids.join(", ") || "None"}</p></div>
        <div><h2>Input modalities</h2><p>{detail.input_modalities.join(", ")}</p></div>
        <div><h2>Scope</h2>{detail.scope ? <Json value={detail.scope} /> : <p>None</p>}</div>
      </section>
      <section className="raw"><h2>Typed targets</h2>
        {detail.targets === null ? <p>Supplied through target annotations.</p>
          : detail.targets.length === 0 ? <p>Explicit empty answer.</p>
          : <ul className="task-targets">{detail.targets.map((target, index) =>
            <li key={index}><span className="task-target-kind">{label(target.kind)}</span>
              {"id" in target ? target.kind === "record"
                ? <button className="task-link" onClick={() => openRecord(target.id)}>{target.id}</button>
                : <code>{target.id}</code>
                : target.kind === "span" ? <Json value={target.value} />
                : <p>{String(target.value)}</p>}
            </li>)}</ul>}
      </section>
      {Array.isArray(detail.configuration.candidate_record_ids) && <section className="raw"><h2>Candidate records</h2>
        {detail.configuration.candidate_record_ids.map((id: string) => <button className="task-link record-task" key={id}
          onClick={() => openRecord(id)}>{id}</button>)}
      </section>}
      <details className="raw"><summary>Task configuration</summary><Json value={detail.configuration} /></details>
      <details className="raw"><summary>Metadata</summary><Json value={detail.metadata} /></details>
      {([ ["Attached task annotations", detail.annotations], ["Input annotations", detail.input_annotations],
        ["Target annotations", detail.target_annotations] ] as const).map(([label, annotations]) =>
        annotations.length > 0 && <details className="raw" key={label}><summary>{label} ({annotations.length})</summary><Json value={annotations} /></details>)}
      <TaskRecordWorkspace key={detail.id} {...props} inputs={detail.input_record_ids}
        targets={detail.targets?.flatMap(target => target.kind === "record" ? [target.id] : []) ?? []}
        candidates={Array.isArray(detail.configuration.candidate_record_ids)
          ? detail.configuration.candidate_record_ids.filter((id): id is string => typeof id === "string") : []} />
    </>}
  </main>;
}
