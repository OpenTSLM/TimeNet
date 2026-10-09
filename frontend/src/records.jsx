import { useEffect, useMemo, useRef, useState } from "react";
import { ComponentPicker, Pagination } from "./components";
import { useCursor, useResource } from "./hooks";
import { JobRun } from "./jobs";
import { progressLabel } from "./api";
import { SignalPlot } from "./plot";

export function Records({ client, scope = null, navigation, returnToTask }) {
  const [query, setQuery] = useState("");
  const [recordId, setRecordId] = useState(navigation?.record_id || null);
  const paging = useCursor(JSON.stringify([query, scope]));
  const records = useResource(client, scope === null ? "/api/v1/records/query" : null,
    { query, after: paging.after, limit: 10 });
  const filtered = scope?.filter((id) => id.toLowerCase().includes(query.toLowerCase()));
  const offset = Number(paging.after || 0);
  const ids = filtered ? filtered.slice(offset, offset + 10) : records.data?.items.map((item) => item.record_id) || [];
  const next = filtered ? (offset + 10 < filtered.length ? String(offset + 10) : null) : records.data?.next_cursor;
  useEffect(() => { if (navigation?.record_id) setRecordId(navigation.record_id); }, [navigation]);
  useEffect(() => { if (!recordId && ids.length) setRecordId(ids[0]); }, [recordId, ids.join("\0")]);
  return <div id="records-workspace" className="layout">
    <aside className="panel"><strong id="records-title">{scope ? "Task input records" : "Records"}</strong>
      <input id="search" placeholder="Search record IDs" value={query} onChange={(event) => setQuery(event.target.value)} />
      <div id="records">{ids.map((id) => <button key={id} data-record-id={id} className={id === recordId ? "selected" : ""}
        onClick={() => setRecordId(id)}>{id}</button>)}<Pagination paging={paging} next={next} /></div>
      {records.error && <p role="alert">{records.error}</p>}
    </aside>
    <section className="panel">{recordId ? <RecordInspector key={`${recordId}:${navigation?.sequence || 0}`}
      client={client} recordId={recordId} target={navigation?.record_id === recordId ? navigation : null}
      returnToTask={returnToTask} /> : <p>Select a record.</p>}</section>
  </div>;
}

function TreePage({ client, recordId, sourceId = null, type, selected, onSelect, onDefaultSelect, defaultSignal = false }) {
  const [after, setAfter] = useState(null);
  const [items, setItems] = useState([]);
  const selectedDefault = useRef(false);
  const path = type === "signals" ? "/api/v1/signals/query" : "/api/v1/sources/query";
  const page = useResource(client, path, { record_id: recordId,
    ...(type === "signals" ? { source_id: sourceId } : { parent_id: sourceId }), after });
  useEffect(() => { if (page.data) setItems((previous) => after ? [...previous, ...page.data.items] : page.data.items); }, [page.data]);
  useEffect(() => {
    if (type === "signals" && defaultSignal && !selectedDefault.current && !selected.length && items.length) {
      selectedDefault.current = true;
      onDefaultSelect(items[0]);
    }
  }, [defaultSignal, items, onDefaultSelect, selected.length, type]);
  return <>{items.map((item, index) => type === "sources" ? <Source key={item.source_id} {...{ client, recordId, selected, onSelect, onDefaultSelect }} source={item}
    defaultOpen={sourceId === null && index === 0} />
    : <button key={item.signal_id} data-signal-id={item.signal_id} className={selected.some((entry) => entry.signal.signal_id === item.signal_id) ? "selected" : ""}
      onClick={(event) => onSelect(item, event.ctrlKey || event.metaKey)}>{item.name} ({item.dtype}, {item.axis_kind})</button>)}
    {page.data?.next_cursor && <button onClick={() => setAfter(page.data.next_cursor)}>More…</button>}
    {page.error && <p role="alert">{page.error}</p>}
  </>;
}

function Source({ client, recordId, source, selected, onSelect, onDefaultSelect, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return <details className="source" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>Source: {source.name}</summary>
    {open && <><TreePage {...{ client, recordId, selected, onSelect, onDefaultSelect }} sourceId={source.source_id} type="signals"
      defaultSignal={defaultOpen} />
      <TreePage {...{ client, recordId, selected, onSelect, onDefaultSelect }} sourceId={source.source_id} type="sources" /></>}
  </details>;
}

function RecordInspector({ client, recordId, target, returnToTask }) {
  const metadata = useResource(client, "/api/v1/records/detail", { record_id: recordId });
  const targetSignal = useResource(client, target?.signal_id ? "/api/v1/records/detail" : null,
    { record_id: recordId, signal_id: target?.signal_id });
  const [selected, setSelected] = useState([]);
  const [picker, setPicker] = useState(null);
  const [error, setError] = useState(null);
  const [start, setStart] = useState("0"), [stop, setStop] = useState("");
  const [utc, setUtc] = useState(false);
  const [window, setWindow] = useState({ start: 0 });
  const [revision, setRevision] = useState(0);
  const clock = metadata.data?.start_time_us == null ? null : BigInt(metadata.data.start_time_us);
  useEffect(() => {
    if (!targetSignal.data) return;
    const spec = targetSignal.data.signal.spec;
    setSelected([{ signal: { signal_id: target.signal_id, name: spec.name, dtype: spec.dtype,
      value_shape: spec.value_shape, axis_kind: null }, component: spec.value_shape.map(() => 0) }]);
  }, [targetSignal.data]);
  useEffect(() => {
    if (target?.start != null) {
      if (target.kind?.startsWith("time")) setWindow({ start_us: target.start, end_us: target.end || String(BigInt(target.start) + 1n) });
      else if (target.kind?.startsWith("step")) {
        const end = target.end || String(BigInt(target.start) + 1n);
        setStart(target.start); setStop(end); setWindow({ start: Number(target.start), stop: Number(end) });
      }
    }
  }, [target]);
  function choose(signal, additive, component = []) {
    setError(null);
    if (additive && selected.length >= 8 && !selected.some((entry) => entry.signal.signal_id === signal.signal_id)) {
      setError("Select at most eight signals."); return;
    }
    setSelected((previous) => {
      if (!additive) return [{ signal, component }];
      return previous.some((entry) => entry.signal.signal_id === signal.signal_id)
        ? previous.filter((entry) => entry.signal.signal_id !== signal.signal_id) : [...previous, { signal, component }];
    });
  }
  function onSelect(signal, additive) {
    if (signal.value_shape.length) setPicker({ signal, additive });
    else choose(signal, additive);
  }
  function selectDefault(signal) { choose(signal, false, signal.value_shape.map(() => 0)); }
  const current = selected.at(-1);
  const request = useMemo(() => ({ record_id: recordId, ...window }), [recordId, window]);
  return <>
    <div id="selection" className="muted">Selected: Record: {recordId}; signals: {selected.length}</div>
    {returnToTask && <button onClick={returnToTask}>Return to task</button>}
    <div id="tensor-notice">{picker && <ComponentPicker key={picker.signal.signal_id} shape={picker.signal.value_shape}
      onCancel={() => setPicker(null)} onChoose={(indexes) => { choose(picker.signal, picker.additive, indexes); setPicker(null); }} />}</div>
    <details><summary>Record metadata</summary><pre id="record-metadata">{JSON.stringify(metadata.data, null, 2)}</pre></details>
    {(error || metadata.error || targetSignal.error) && <p role="alert">{error || metadata.error || targetSignal.error}</p>}
    <div id="sources"><TreePage {...{ client, recordId, selected, onSelect }} onDefaultSelect={selectDefault} type="sources" /></div>
    <div className="controls"><label>Start <input id="start" type="number" min="0" value={start} onChange={(e) => setStart(e.target.value)} /></label>
      <label>Stop <input id="stop" type="number" min="1" value={stop} onChange={(e) => setStop(e.target.value)} /></label>
      <button id="window" onClick={() => { setWindow({ start: Number(start), ...(stop ? { stop: Number(stop) } : {}) }); setRevision((v) => v + 1); }}>Update window</button>
      <button id="fit" onClick={() => { setStart("0"); setStop(""); setWindow({ start: 0, full: true }); setRevision((v) => v + 1); }}>Fit signal</button>
      <button id="reset" onClick={() => { setSelected([]); setPicker(null); setWindow({ start: 0 }); setStart("0"); setStop(""); setRevision((v) => v + 1); }}>Reset</button>
    </div>
    <p id="window-cost" className="muted">Fit signal reads the entire signal for each selected component, even when the plot is reduced.
      {current?.signal.n_values != null && ` Selected signal: ${current.signal.n_values} steps.`}</p>
    <label><input id="utc" type="checkbox" disabled={clock === null} checked={utc} onChange={(event) => setUtc(event.target.checked)} />UTC timestamps (when known)</label>
    {selected.length ? <WindowPanel key={JSON.stringify([request, selected, revision])} {...{ client, selected, request, clock, utc }}
      onViewport={(first, last, kind, origin) => {
        if (kind === "ordinal") { setStart(String(Math.max(0, Math.floor(first)))); setStop(String(Math.ceil(last))); setWindow({ start: Math.max(0, Math.floor(first)), stop: Math.ceil(last) }); }
        else setWindow({ start_us: String(origin + BigInt(Math.floor(first))), end_us: String(origin + BigInt(Math.ceil(last))) });
      }} /> : <><div id="plot" /><div id="raw" /></>}
  </>;
}

function WindowPanel({ client, selected, request, clock, utc, onViewport }) {
  const [windows, setWindows] = useState([]), [errors, setErrors] = useState([]), [progress, setProgress] = useState({});
  const current = selected.at(-1);
  function forSignal(entry) {
    const query = { ...request, signal_id: entry.signal.signal_id, component: entry.component };
    if (entry.signal.axis_kind === "ordinal") { delete query.start_us; delete query.end_us; }
    return query;
  }
  const rawRequest = { ...forSignal(current), mode: "raw" };
  const [annotationCursor, setAnnotationCursor] = useState(null), [annotations, setAnnotations] = useState([]);
  const overlay = useResource(client, "/api/v1/annotations/window", { record_id: request.record_id, signal_id: current.signal.signal_id,
    start_us: request.start_us, end_us: request.end_us, after: annotationCursor });
  useEffect(() => { if (overlay.data) setAnnotations((items) => annotationCursor ? [...items, ...overlay.data.items] : overlay.data.items); }, [overlay.data]);
  useEffect(() => {
    let active = true;
    const entries = selected.filter((entry) => !["str", "bool", "enum"].includes(entry.signal.dtype));
    const runs = entries.map(() => new JobRun(client));
    Promise.allSettled(runs.map((run, i) => run.start("/api/v1/jobs/windows", { ...forSignal(entries[i]), mode: "plot", width: 800 },
      (job) => { if (active) setProgress((old) => ({ ...old, [i]: job })); }))).then((results) => {
      if (!active) return;
      setWindows(results.filter((item) => item.status === "fulfilled").map((item) => item.value));
      setErrors(results.filter((item) => item.status === "rejected").map((item) => item.reason.message));
    });
    return () => { active = false; runs.forEach((run) => run.dispose()); };
  }, [client]);
  return <>
    <div id="axis-label" className="muted">{windows.some((data) => data.reduced)
      ? "Reduced first/last/extrema markers; coverage counts describe every scanned sample."
      : "Regular signals use broken lines at missing samples; irregular signals use markers."}</div>
    <div id="window-progress" aria-live="polite">{Object.entries(progress).map(([id, job]) => <p key={id}>{progressLabel(job)}</p>)}</div>
    <SignalPlot {...{ windows, annotations, clock, utc, onViewport }} />
    <div id="coverage">{windows.map((data) => <div className="coverage-summary" key={data.signal_id}>
      <strong>{data.signal_name}: {data.scanned_count} samples</strong>
      {Object.entries(data.coverage).filter(([kind, count]) => kind !== "finite" && BigInt(count) > 0n).map(([kind, count]) =>
        <span className="coverage-badge" key={kind}>{count} {kind}</span>)}
    </div>)}</div>
    {[...errors, overlay.error].filter(Boolean).map((error, i) => <p role="alert" key={i}>{error}</p>)}
    <div id="plot-annotations">{annotations.map((item) => <p key={JSON.stringify([item.dataset_id, item.version, item.occurrence_id])}>{item.object_type}: {item.name} ({item.span_type})</p>)}
      {overlay.data?.next_cursor && <button onClick={() => setAnnotationCursor(overlay.data.next_cursor)}>More annotations</button>}</div>
    <RawValues client={client} request={rawRequest} />
  </>;
}

function RawValues({ client, request }) {
  const [bounds, setBounds] = useState(null);
  const page = useResource(client, "/api/v1/windows", { ...request, ...bounds });
  const [first, setFirst] = useState(null);
  useEffect(() => { if (page.data && first === null) setFirst(Number(page.data.requested.start)); }, [page.data]);
  const data = page.data;
  function move(start) { setBounds({ start, stop: Number(data.requested.stop), start_us: null, end_us: null }); }
  return <div id="raw">{page.error && <p role="alert">{page.error}</p>}{data && <>
    <details key={data.requested.start}><summary>Per-step values ({data.items.length} displayed)</summary>
      <table><thead><tr><th>Step</th><th>Time / step</th><th>Value</th></tr></thead><tbody>
        {data.items.map((item) => <tr key={item.index}><td>{item.index}</td><td>{item.x}</td><td>{item.display ?? "null"}</td></tr>)}
      </tbody></table></details>
    {Number(data.requested.start) > first && <button onClick={() => move(Math.max(first, Number(data.requested.start) - 200))}>Previous values</button>}
    {data.next_start !== null && <button onClick={() => move(Number(data.next_start))}>Next values</button>}
  </>}</div>;
}
