import { useEffect, useState } from "react";
import { changeSelection, inspectorHref, recordChoices, returnFromInspector, selectRecord } from "./navigation";
import type { RecordChoice, ViewerProps } from "./navigation";

// Steps loaded when a signal is selected, before the user picks a window.
const INITIAL_WINDOW = 1000;

type Source = { id: string; name: string; parent_source_id: string | null; metadata: unknown; annotations: unknown[] };
type Signal = {
  id: string; source_id: string; name: string; n_values: number; metadata: unknown;
  annotations: unknown[]; spec: { dtype: string; unit_value: string | null }; axis: unknown;
  plot_supported: boolean; plot_reason: string | null;
};
type Detail = {
  id: string; start_time: number | null; time_span: unknown; metadata: unknown;
  annotations: unknown[]; sources: Source[]; signals: Signal[];
};
type Window = { start: number; stop: number; values: (number | null)[]; nonfinite_steps: number[] };

function Facts({ title, value }: { title: string; value: unknown }) {
  return <details className="inspector-facts"><summary>{title}</summary><pre className="task-json">{JSON.stringify(value, null, 2)}</pre></details>;
}

function ScalarPlot({ window, unit }: { window: Window; unit: string | null }) {
  const finite = window.values.filter((v): v is number => v !== null);
  if (!finite.length) return <p className="empty">This window contains only missing values.</p>;
  const low = Math.min(...finite), high = Math.max(...finite);
  // Normalize first to keep finite float extremes and constant signals plottable.
  const scale = Math.max(Math.abs(low), Math.abs(high), 1);
  const y = (value: number) => low === high ? 130
    : 230 - ((value / scale - low / scale) / (high / scale - low / scale)) * 200;
  const x = (index: number) => 75 + index / Math.max(1, window.values.length - 1) * 600;
  let path = "", connected = false;
  window.values.forEach((value, index) => {
    if (value === null) { connected = false; return; }
    path += `${connected ? "L" : "M"}${x(index)},${y(value)} `;
    connected = true;
  });
  return <svg className="scalar-plot" viewBox="0 0 720 290" role="img"
    aria-label={`Signal values from step ${window.start} to ${window.stop - 1}, ${unit || "unit unknown"}`}>
    <path d="M75 20V230H680" fill="none" stroke="currentColor" />
    <path d={path} fill="none" stroke="#26a69a" strokeWidth="2" />
    {window.values.map((v, i) => v !== null && <circle key={i} cx={x(i)} cy={y(v)} r="1.5" fill="#26a69a" />)}
    <text x="70" y="30" textAnchor="end">{high.toPrecision(4)}</text>
    <text x="70" y="230" textAnchor="end">{low.toPrecision(4)}</text>
    <text x="75" y="250">{window.start}</text><text x="675" y="250" textAnchor="end">{window.stop - 1}</text>
    <text x="75" y="14">Value · {unit || "unit unknown"}</text>
    <text x="375" y="275" textAnchor="middle">Step index</text>
  </svg>;
}

function SignalInspector({ selected, recordId, ...props }: ViewerProps & { selected: Signal; recordId: string }) {
  const params = new URLSearchParams(props.search);
  const submittedStart = params.get("start"), submittedStop = params.get("stop");
  const [start, setStart] = useState(submittedStart ?? "0");
  const [stop, setStop] = useState(submittedStop ?? String(Math.min(INITIAL_WINDOW, selected.n_values)));
  const [window, setWindow] = useState<Window | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    setStart(submittedStart ?? "0");
    setStop(submittedStop ?? String(Math.min(INITIAL_WINDOW, selected.n_values)));
    const controller = new AbortController();
    setWindow(null); setError(null); setLoading(false);
    if (submittedStart === null || submittedStop === null || !selected.plot_supported || selected.n_values === 0)
      return () => controller.abort();
    setLoading(true);
    const query = new URLSearchParams({ dataset_id: props.dataset, version: props.version, record_id: recordId,
      signal_id: selected.id, start: submittedStart, stop: submittedStop });
    props.getJson<Window>(`/api/v1/signal-window?${query}`, controller.signal).then(value => {
      if (!controller.signal.aborted) setWindow(value);
    }).catch(reason => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Window request failed.");
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [props.dataset, props.version, props.getJson, recordId, selected.id, selected.n_values,
    selected.plot_supported, submittedStart, submittedStop, retry]);
  function load(first: string, last: string) {
    const next = changeSelection(props.search, { start: first, stop: last });
    props.navigate(next); setRetry(value => value + 1);
  }
  const width = Number(stop) - Number(start);
  const valid = Number.isSafeInteger(Number(start)) && Number.isSafeInteger(Number(stop)) &&
    Number(start) >= 0 && width > 0 && Number(stop) <= selected.n_values;
  return <section className="raw" aria-label="Signal inspector">
    <div className="signal-heading"><span className="node-kind node-kind-signal">Signal</span><h2>{selected.name}</h2></div>
    <p className="signal-summary">{selected.n_values.toLocaleString()} steps · {selected.spec.dtype} · {selected.spec.unit_value || "Unit unknown"}</p>
    <Facts title="Signal metadata, spec, axis, and annotations" value={selected} />
    {!selected.plot_supported ? <p>{selected.plot_reason}</p>
      : selected.n_values === 0 ? <p className="empty">This signal has no values.</p> : <>
        <form className="filters" onSubmit={event => { event.preventDefault(); load(start, stop); }}>
          <label>Start step<input type="number" min="0" step="1" required value={start} onChange={e => setStart(e.target.value)} /></label>
          <label>Stop step (exclusive)<input type="number" min="1" step="1" required value={stop} onChange={e => setStop(e.target.value)} /></label>
          <button>Load window</button>
        </form>
        <nav className="pagination" aria-label="Signal windows">
          <button disabled={!valid || Number(start) === 0} onClick={() => {
            const first = Math.max(0, Number(start) - width); load(String(first), String(first + width));
          }}>Previous window</button>
          <button disabled={!valid || Number(stop) >= selected.n_values} onClick={() =>
            load(stop, String(Math.min(selected.n_values, Number(stop) + width)))}>Next window</button>
        </nav>
        {loading && <p role="status">Loading window…</p>}
        {error && <div role="alert"><p>{error}</p><button onClick={() => setRetry(v => v + 1)}>Retry window</button></div>}
        {window && <><ScalarPlot window={window} unit={selected.spec.unit_value} />
          <p>Missing values appear as gaps.{window.nonfinite_steps.length > 0 &&
            ` Nonfinite values also appear as gaps at steps ${window.nonfinite_steps.join(", ")}.`}</p></>}
      </>}
  </section>;
}

function RecordSelector({ choices, selected, onSelect }: {
  choices: RecordChoice[]; selected: string | null; onSelect: (id: string) => void;
}) {
  return <aside className="record-selector" aria-label="Task records">
    <h3>Records</h3>
    {!choices.some(choice => choice.roles.includes("Input")) && <p>No input records.</p>}
    {choices.map(choice => <div className="record-choice" key={choice.id}>
      <button className="task-link" aria-pressed={choice.id === selected} onClick={() => onSelect(choice.id)}>
        {choice.id}<small>{choice.roles.join(" · ")}</small>
      </button>
    </div>)}
  </aside>;
}

export function TaskRecordWorkspace({ inputs, targets, candidates, ...props }: ViewerProps & {
  inputs: string[]; targets: string[]; candidates: string[];
}) {
  const choices = recordChoices(inputs, targets, candidates);
  const requested = new URLSearchParams(props.search).get("record");
  const recordId = choices.some(choice => choice.id === requested) ? requested : inputs[0] ?? null;
  useEffect(() => {
    if (recordId !== requested) {
      const next = requested ? selectRecord(props.search, recordId ?? "") : changeSelection(props.search, { record: recordId });
      props.navigate(next, { replace: true });
    }
  }, [recordId, requested, props.search, props.navigate]);
  return <section className="raw" aria-label="Task record workspace">
    <h2>Record workspace</h2>
    <div className="record-workspace">
      <RecordSelector choices={choices} selected={recordId}
        onSelect={id => props.navigate(selectRecord(props.search, id))} />
      {recordId ? <RecordInspector key={recordId} {...props} recordId={recordId} />
        : <p className="empty">Select a referenced record to inspect its data.</p>}
    </div>
  </section>;
}

type SourceBranchProps = {
  source: Source; childrenBySource: Map<string | null, Source[]>; signalsBySource: Map<string, Signal[]>;
  expanded: Set<string>; selectedSource: string | null; selectedSignal: string | null; root?: boolean;
  inspectSource: (source: Source) => void; inspectSignal: (signal: Signal) => void;
};

function SourceBranch({ source, root = false, ...props }: SourceBranchProps) {
  const shouldExpand = props.expanded.has(source.id);
  const [open, setOpen] = useState(root || shouldExpand);
  useEffect(() => { if (shouldExpand) setOpen(true); }, [shouldExpand]);
  const children = props.childrenBySource.get(source.id) ?? [];
  const signals = props.signalsBySource.get(source.id) ?? [];
  return <li className="hierarchy-source">
    <details open={open} onToggle={event => setOpen(event.currentTarget.open)}>
      <summary aria-label={`Source ${source.name} (${source.id})`}>
        <span className="node-kind node-kind-source">Source</span>
        <strong title={source.id}>{source.name}</strong>
        <small>{signals.length} {signals.length === 1 ? "signal" : "signals"}{children.length > 0 &&
          ` · ${children.length} ${children.length === 1 ? "source" : "sources"}`}</small>
      </summary>
      <div className="hierarchy-source-actions">
        <button className="task-link" aria-label={`Inspect source ${source.name}`}
          aria-pressed={source.id === props.selectedSource} onClick={() => props.inspectSource(source)}>Source details</button>
      </div>
      <ul className="hierarchy-children">
        {signals.map(signal => <li className="hierarchy-signal" key={signal.id}>
          <button aria-pressed={signal.id === props.selectedSignal} aria-label={`Signal ${signal.name} (${signal.id})`}
            title={signal.id} onClick={() => props.inspectSignal(signal)}>
            <span className="signal-glyph" aria-hidden="true">∿</span>
            <span className="node-kind node-kind-signal">Signal</span>
            <strong>{signal.name}</strong>
            <small>{signal.n_values.toLocaleString()} steps</small>
            {signal.id === props.selectedSignal && <span className="selected-node-label">Selected</span>}
          </button>
        </li>)}
        {children.map(child => <SourceBranch key={child.id} {...props} source={child} />)}
        {!signals.length && !children.length && <li className="hierarchy-empty">No signals or child sources.</li>}
      </ul>
    </details>
  </li>;
}

function SourceInspector({ detail, ...props }: ViewerProps & { detail: Detail }) {
  const params = new URLSearchParams(props.search);
  const sourceId = params.get("source"), signalId = params.get("signal");
  const sourceIds = new Set(detail.sources.map(source => source.id));
  const selectedSignal = detail.signals.find(signal => signal.id === signalId && sourceIds.has(signal.source_id));
  const selectedSource = detail.sources.find(source => source.id === (selectedSignal?.source_id ?? sourceId));
  useEffect(() => {
    if (selectedSignal && sourceId !== selectedSignal.source_id) {
      props.navigate(changeSelection(props.search, { source: selectedSignal.source_id }), { replace: true });
    }
  }, [selectedSignal?.id, selectedSignal?.source_id, sourceId, props.search, props.navigate]);
  const childrenBySource = new Map<string | null, Source[]>(), signalsBySource = new Map<string, Signal[]>();
  for (const source of detail.sources) {
    const children = childrenBySource.get(source.parent_source_id) ?? [];
    children.push(source); childrenBySource.set(source.parent_source_id, children);
  }
  for (const signal of detail.signals) {
    if (!sourceIds.has(signal.source_id)) continue;
    const signals = signalsBySource.get(signal.source_id) ?? [];
    signals.push(signal); signalsBySource.set(signal.source_id, signals);
  }
  const path: Source[] = [], expanded = new Set<string>();
  let ancestor = detail.sources.find(source => source.id === selectedSignal?.source_id) ?? selectedSource;
  while (ancestor && !expanded.has(ancestor.id)) {
    path.unshift(ancestor); expanded.add(ancestor.id);
    ancestor = detail.sources.find(source => source.id === ancestor!.parent_source_id);
  }
  function go(changes: Record<string, string | null>) {
    props.navigate(changeSelection(props.search, changes));
  }
  const roots = childrenBySource.get(null) ?? [];
  return <section aria-label="Sources and signals">
    <nav className="signal-hierarchy" aria-label="Record hierarchy">
      {roots.length ? <ul className="hierarchy-roots">{roots.map(source =>
        <SourceBranch key={source.id} source={source} root expanded={expanded}
          childrenBySource={childrenBySource} signalsBySource={signalsBySource}
          selectedSource={selectedSource?.id ?? null} selectedSignal={signalId}
          inspectSource={item => go({ source: item.id, signal: null, start: null, stop: null })}
          inspectSignal={item => go({ source: item.source_id, signal: item.id,
            start: item.plot_supported && item.n_values > 0 ? "0" : null,
            stop: item.plot_supported && item.n_values > 0 ? String(Math.min(INITIAL_WINDOW, item.n_values)) : null })}
        />)}</ul> : <p>No sources.</p>}
    </nav>
    {sourceId && !selectedSource && <p role="alert">Source not found in this record.</p>}
    {signalId && !selectedSignal && <p role="alert">Signal not found in this record.</p>}
    {selectedSignal ? <>
      <nav className="signal-path" aria-label="Selected signal path"><ol>
        <li><span className="node-kind node-kind-record">Record</span>{detail.id}</li>
        {path.map(source => <li key={source.id}><span className="path-separator" aria-hidden="true">›</span>
          <button className="task-link" title={source.id}
            onClick={() => go({ source: source.id, signal: null, start: null, stop: null })}>{source.name}</button></li>)}
        <li aria-current="location"><span className="path-separator" aria-hidden="true">›</span>{selectedSignal.name}</li>
      </ol></nav>
      <SignalInspector key={`${props.dataset}@${props.version}:${detail.id}:${selectedSignal.id}`}
        {...props} selected={selectedSignal} recordId={detail.id} />
    </> : !signalId && <div className="signal-empty"><span className="signal-glyph" aria-hidden="true">∿</span>
      <div><h3>Select a signal</h3><p>Choose a signal in the hierarchy above to inspect it and load a value window.</p></div>
    </div>}
    {selectedSource && <Facts title={`Source details: ${selectedSource.name}`} value={selectedSource} />}
  </section>;
}

function RecordInspector({ recordId, ...props }: ViewerProps & { recordId: string }) {
  const { dataset, version, getJson } = props;
  const [result, setResult] = useState<{ identity: string; detail?: Detail; error?: string } | null>(null);
  const [retry, setRetry] = useState(0);
  const identity = `${dataset}@${version}:${recordId}`;
  useEffect(() => {
    const controller = new AbortController();
    setResult(null);
    const query = new URLSearchParams({ dataset_id: dataset, version, record_id: recordId });
    getJson<Detail>(`/api/v1/record?${query}`, controller.signal).then(detail => {
      if (!controller.signal.aborted) setResult({ identity, detail });
    }).catch(reason => {
      if (!controller.signal.aborted) setResult({ identity, error: reason instanceof Error ? reason.message : "Record request failed." });
    });
    return () => controller.abort();
  }, [dataset, version, recordId, identity, retry, getJson]);
  const current = result?.identity === identity ? result : null;
  return <section className="record-inspector" aria-label="Record inspector">
    {new URLSearchParams(props.search).get("view") === "tasks" &&
      <div className="inspector-links"><h3>{recordId}</h3>
        <a href={inspectorHref(props.search, recordId)} target="_blank" rel="opener" aria-label={`Open record ${recordId} in record view`}>Open record view ↗</a>
      </div>}
    {current?.error ? <div role="alert"><p>{current.error}</p><button onClick={() => setRetry(v => v + 1)}>Retry record</button></div>
      : !current?.detail && <p role="status">Preparing record data… Remote versions may take time to download.</p>}
    {current?.detail && <>
      <SourceInspector {...props} detail={current.detail} />
      <Facts title="Record facts, metadata, and annotations" value={{
        start_time: current.detail.start_time, time_span: current.detail.time_span,
        metadata: current.detail.metadata, annotations: current.detail.annotations,
      }} />
    </>}
  </section>;
}

export function RecordsBrowser(props: ViewerProps) {
  const params = new URLSearchParams(props.search);
  const recordId = params.get("record");
  return <main className="layout task-layout">
    <a className="back" href={`/?${returnFromInspector(props.search)}`} onClick={event => {
      if (event.button === 0 && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey) {
        event.preventDefault(); props.navigate(returnFromInspector(props.search));
      }
    }}>← {params.get("task") ? "Return to task" : "Task list"}</a>
    <header><p className="eyebrow">TimeNet viewer · Record details</p>
      <h1>{recordId || "Record details"}</h1>
      <p className="identity">{props.dataset} · {props.version}</p>
    </header>
    {!recordId ? <p role="alert">Supply a record ID to inspect its data.</p>
      : <RecordInspector key={recordId} {...props} recordId={recordId} />}
  </main>;
}
