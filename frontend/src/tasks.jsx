import { useEffect, useState } from "react";
import { Pagination } from "./components";
import { useCursor, useResource } from "./hooks";
import { Records } from "./records";
import { annotationLabel } from "./annotations";

export function Tasks({ client, types, taskId, selectTask, navigate, openAnnotation }) {
  const [query, setQuery] = useState(""), [type, setType] = useState(""), [split, setSplit] = useState("");
  const paging = useCursor(JSON.stringify([query, type, split]));
  const page = useResource(client, "/api/v1/tasks/query", { query, task_type: type || null, split: split || null, after: paging.after, limit: 10 });
  useEffect(() => {
    if (!taskId && page.data?.items.length) selectTask(page.data.items[0].task_id);
  }, [page.data, selectTask, taskId]);
  return <>
    <section className="panel"><h2>Tasks</h2>
      <input id="task-search" placeholder="Search task IDs" value={query} onChange={(e) => setQuery(e.target.value)} />
      <select id="task-type" value={type} onChange={(e) => setType(e.target.value)}><option value="">All task types</option>{types.map((type) => <option key={type}>{type}</option>)}</select>
      <select id="task-split" value={split} onChange={(e) => setSplit(e.target.value)}><option value="">All splits</option>
        {["train", "validation", "test"].map((name) => <option key={name}>{name}</option>)}<option value="__unassigned__">Unassigned</option></select>
      <div id="task-list">{page.data?.items.map((task) => <button key={task.task_id} data-task-id={task.task_id}
        className={taskId === task.task_id ? "selected" : ""} onClick={() => selectTask(task.task_id)}>
        {task.task_id} — {task.task_type}{task.prompt ? `: ${task.prompt}` : ""}</button>)}
        <Pagination paging={paging} next={page.data?.next_cursor} /></div>
      {page.error && <p role="alert">{page.error}</p>}
    </section>
    <section className="panel"><h2>Task context</h2><div id="task-selection" className="muted">{taskId ? `Selected: Task: ${taskId}` : "Select a task."}</div>
      {taskId && <TaskDetail key={taskId} {...{ client, taskId, selectTask, navigate, openAnnotation }} />}
    </section>
  </>;
}

function Section({ title, items, children }) {
  return items.length ? <><h4>{title}</h4><div className={title === "Targets" ? "target-grid" : ""}>{items.map(children)}</div></> : null;
}

function TargetCard({ target, index, navigate }) {
  const titles = { boolean: "Boolean answer", text: "Text answer", integer: "Integer answer", float: "Numeric answer",
    record: "Record reference", signal: "Signal reference", time_point: "Time point", time_interval: "Time interval", step_point: "Step point", step_interval: "Step interval" };
  const fields = [];
  if (target.value !== null) fields.push(["Answer", String(target.value)]);
  if (target.record_id) fields.push(["Record", target.record_id]);
  if (target.signal_id) fields.push(["Signal", target.signal_id]);
  if (target.start !== null) fields.push(["Span", `[${target.start}, ${target.end ?? "…"})`]);
  if (target.signal_ids.length) fields.push(["Signals", target.signal_ids.join(", ")]);
  return <article className="target-card"><h5>Target {index + 1}: {titles[target.kind] || "Target"}</h5>
    <dl>{fields.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
    {(target.record_id || target.signal_id || target.signal_ids.length || target.start !== null) && <button onClick={() => navigate(target)}>Inspect target</button>}
  </article>;
}

function TaskDetail({ client, taskId, selectTask, navigate, openAnnotation }) {
  const [after, setAfter] = useState(0);
  const result = useResource(client, "/api/v1/tasks/detail", { task_id: taskId, after });
  const task = result.data;
  const go = (target) => navigate(target, task);
  return <><div id="task-detail">{result.error && <p role="alert">{result.error}</p>}{task && <>
    <h3>{task.task_id} — {task.task_type}</h3>
    {[task.prompt, task.rationale, task.split && `split: ${task.split}`].filter(Boolean).map((text, i) => <p key={i}>{text}</p>)}
    <Section title="Targets" items={task.targets}>{(target, i) => <TargetCard key={i} target={target} index={after + i} navigate={go} />}</Section>
    <pre>{JSON.stringify({ scope: task.scope, configuration: task.configuration, metadata: task.metadata,
      inline_targets: task.has_inline_targets ? (task.targets.length ? "present" : "empty on this page") : "not supplied",
      prompt_truncated: task.prompt_truncated, rationale_truncated: task.rationale_truncated }, null, 2)}</pre>
    <Section title="Candidate records" items={task.candidate_record_ids}>{(id) => <button key={id} onClick={() => go({ record_id: id })}>{id}</button>}</Section>
    {["input_annotations", "target_annotations"].map((field) => <Section key={field} title={field === "input_annotations" ? "Input annotations" : "Target annotations"}
      items={task.annotation_refs.filter((item) => item.field === field)}>{(item) => <button key={item.occurrence_id} onClick={() => openAnnotation(item)}>{annotationLabel(item)}</button>}</Section>)}
    <Section title="Parent tasks" items={task.parent_task_ids}>{(id) => <button key={id} onClick={() => selectTask(id)}>{id}</button>}</Section>
    {after > 0 && <button onClick={() => setAfter(Math.max(0, after - 200))}>Previous relationships</button>}
    {task.next_after !== null && <button onClick={() => setAfter(task.next_after)}>Next relationships</button>}
  </>}</div><div id="task-record-context">{task?.input_record_ids.length > 0 && <Records key={`${taskId}:${after}`} client={client} scope={task.input_record_ids} />}</div></>;
}
