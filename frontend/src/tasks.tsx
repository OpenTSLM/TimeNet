import type { ReactNode } from "react";
import type { Schemas } from "./api";
import type { ClientProps, Navigate, OpenAnnotation, Target } from "./types";
interface TaskProps extends ClientProps {
  taskId: string | null;
  selectTask: (id: string) => void;
  navigate: Navigate;
  openAnnotation: OpenAnnotation;
}
import { useEffect, useState } from "react";
import { Pagination } from "./components";
import { useCursor, useResource } from "./hooks";
import { Records } from "./records";
import { annotationLabel } from "./annotations";

export function Tasks({
  client,
  types,
  taskId,
  selectTask,
  navigate,
  openAnnotation,
}: TaskProps & { types: string[] }) {
  const [query, setQuery] = useState(""),
    [type, setType] = useState(""),
    [split, setSplit] = useState("");
  const paging = useCursor(JSON.stringify([query, type, split]));
  const page = useResource(client, "POST /api/v1/tasks/query", {
    query,
    task_type: type || null,
    split: split || null,
    after: paging.after,
    limit: 10,
  });
  useEffect(() => {
    if (!taskId && page.data?.items.length)
      selectTask(page.data.items[0].task_id);
  }, [page.data, selectTask, taskId]);
  return (
    <>
      <section className="panel">
        <h2>Tasks</h2>
        <input
          id="task-search"
          placeholder="Search task IDs"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select
          id="task-type"
          value={type}
          onChange={(e) => setType(e.target.value)}
        >
          <option value="">All task types</option>
          {types.map((type) => (
            <option key={type}>{type}</option>
          ))}
        </select>
        <select
          id="task-split"
          value={split}
          onChange={(e) => setSplit(e.target.value)}
        >
          <option value="">All splits</option>
          {["train", "validation", "test"].map((name) => (
            <option key={name}>{name}</option>
          ))}
          <option value="__unassigned__">Unassigned</option>
        </select>
        <div id="task-list">
          {page.data?.items.map((task) => (
            <button
              key={task.task_id}
              data-task-id={task.task_id}
              className={taskId === task.task_id ? "selected" : ""}
              onClick={() => selectTask(task.task_id)}
            >
              {task.task_id} — {task.task_type}
              {task.prompt ? `: ${task.prompt}` : ""}
            </button>
          ))}
          <Pagination paging={paging} next={page.data?.next_cursor} />
        </div>
        {page.error && <p role="alert">{page.error}</p>}
      </section>
      <section className="panel">
        <h2>Task context</h2>
        <div id="task-selection" className="muted">
          {taskId ? `Selected: Task: ${taskId}` : "Select a task."}
        </div>
        {taskId && (
          <TaskDetail
            key={taskId}
            {...{ client, taskId, selectTask, navigate, openAnnotation }}
          />
        )}
      </section>
    </>
  );
}

function Section<T>({
  title,
  items,
  children,
}: {
  title: string;
  items: T[];
  children: (item: T, index: number) => ReactNode;
}) {
  return items.length ? (
    <>
      <h4>{title}</h4>
      <div className={title === "Targets" ? "target-grid" : ""}>
        {items.map(children)}
      </div>
    </>
  ) : null;
}

function TargetCard({
  target,
  index,
  navigate,
}: {
  target: Schemas["TargetItem"];
  index: number;
  navigate: Navigate;
}) {
  const fields = [];
  if (target.value !== null) fields.push(["Answer", String(target.value)]);
  if (target.record_id) fields.push(["Record", target.record_id]);
  if (target.signal_id) fields.push(["Signal", target.signal_id]);
  if (target.start !== null)
    fields.push(["Span", `[${target.start}, ${target.end ?? "…"})`]);
  if (target.signal_ids.length)
    fields.push(["Signals", target.signal_ids.join(", ")]);
  return (
    <article className="target-card">
      <h5>
        Target {index + 1}: {targetTitle(target)}
      </h5>
      {target.value_truncated && <p>Value preview truncated.</p>}
      {target.signal_ids_truncated && (
        <p>Showing the first 200 signal references.</p>
      )}
      <dl>
        {fields.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {(target.record_id ||
        target.signal_id ||
        target.signal_ids.length ||
        target.start !== null) && (
        <button onClick={() => navigate(target)}>Inspect target</button>
      )}
    </article>
  );
}

function TaskDetail({
  client,
  taskId,
  selectTask,
  navigate,
  openAnnotation,
}: TaskProps & { taskId: string }) {
  const [after, setAfter] = useState(0);
  const result = useResource(client, "POST /api/v1/tasks/detail", {
    task_id: taskId,
    after,
  });
  const task = result.data;
  const go = (target: Target) => navigate(target, task);
  return (
    <>
      <div id="task-detail">
        {result.error && <p role="alert">{result.error}</p>}
        {task && (
          <>
            <h3>
              {task.task_id} — {task.task_type}
            </h3>
            {[task.prompt, task.rationale, task.split && `split: ${task.split}`]
              .filter(Boolean)
              .map((text, i) => (
                <p key={i}>{text}</p>
              ))}
            <Section title="Targets" items={task.targets}>
              {(target, i) => (
                <TargetCard
                  key={i}
                  target={target}
                  index={after + i}
                  navigate={go}
                />
              )}
            </Section>
            <pre>
              {JSON.stringify(
                {
                  scope: task.scope,
                  configuration: task.configuration,
                  inline_targets: task.has_inline_targets
                    ? task.targets.length
                      ? "present"
                      : "empty on this page"
                    : "not supplied",
                  prompt_truncated: task.prompt_truncated,
                  rationale_truncated: task.rationale_truncated,
                },
                null,
                2,
              )}
            </pre>
            <details>
              <summary>
                Task metadata
                {task.metadata_truncated ? " (truncated preview)" : ""}
              </summary>
              <pre>{task.metadata}</pre>
            </details>
            <Section
              title="Candidate records"
              items={task.candidate_record_ids}
            >
              {(id) => (
                <button key={id} onClick={() => go({ record_id: id })}>
                  {id}
                </button>
              )}
            </Section>
            {["input_annotations", "target_annotations"].map((field) => (
              <Section
                key={field}
                title={
                  field === "input_annotations"
                    ? "Input annotations"
                    : "Target annotations"
                }
                items={task.annotation_refs.filter(
                  (item) => item.field === field,
                )}
              >
                {(item) => (
                  <button
                    key={item.occurrence_id}
                    onClick={() => openAnnotation(item)}
                  >
                    {annotationLabel(item)}
                  </button>
                )}
              </Section>
            ))}
            <Section title="Parent tasks" items={task.parent_task_ids}>
              {(id) => (
                <button key={id} onClick={() => selectTask(id)}>
                  {id}
                </button>
              )}
            </Section>
            {after > 0 && (
              <button onClick={() => setAfter(Math.max(0, after - 200))}>
                Previous relationships
              </button>
            )}
            {task.next_after !== null && (
              <button onClick={() => setAfter(task.next_after!)}>
                Next relationships
              </button>
            )}
          </>
        )}
      </div>
      <div id="task-record-context">
        {task && task.input_record_ids.length > 0 && (
          <Records
            key={`${taskId}:${after}`}
            client={client}
            scope={task.input_record_ids}
          />
        )}
      </div>
    </>
  );
}

function targetTitle(target: Schemas["TargetItem"]): string {
  switch (target.kind) {
    case "boolean":
      return "Boolean answer";
    case "text":
      return "Text answer";
    case "integer":
      return "Integer answer";
    case "float":
      return "Numeric answer";
    case "record":
      return "Record reference";
    case "signal":
      return "Signal reference";
    case "time_point":
      return "Time point";
    case "time_interval":
      return "Time interval";
    case "step_point":
      return "Step point";
    case "step_interval":
      return "Step interval";
    default:
      return unreachableTarget(target);
  }
}

function unreachableTarget(target: never): never {
  throw new Error(`Unsupported target: ${JSON.stringify(target)}`);
}
