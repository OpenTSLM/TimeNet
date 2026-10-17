import type {
  Annotation,
  CatalogItem,
  Navigation,
  Target,
  Task,
} from "./types";
import { createRoot } from "react-dom/client";
import { useRef, useState } from "react";
import { ApiClient, errorMessage, type Schemas } from "./api";
import { useConnectionStatus, useResource } from "./hooks";
import { Overview } from "./overview";
import { Records } from "./records";
import { Tasks } from "./tasks";
import { Annotations } from "./annotations";
import { Catalog } from "./catalog";

const token = new URLSearchParams(location.hash.slice(1)).get("token");
history.replaceState(null, "", location.pathname);
const client = new ApiClient(token || "");

function App() {
  const [sessionGeneration, setSessionGeneration] = useState(0),
    [mode, setMode] = useState<"registry" | "dataset" | null>(null);
  const connection = useConnectionStatus(client);
  const session = useResource(
    client,
    token ? "GET /api/v1/session" : null,
    undefined,
    { query: { generation: String(sessionGeneration) } },
  );
  const activeMode = session.data?.mode || mode;
  const overview = useResource(
    client,
    session.data && activeMode !== "registry" ? "GET /api/v1/overview" : null,
    undefined,
  );
  const [catalogAfter, setCatalogAfter] = useState("");
  const [catalogFilters, setCatalogFilters] = useState<Schemas["CatalogQuery"]>(
    {},
  );
  const catalog = useResource(
    client,
    activeMode === "registry" ? "GET /api/v1/registry/datasets" : null,
    undefined,
    { query: { ...catalogFilters, after: catalogAfter } },
  );
  const [view, setView] = useState("overview"),
    [error, setError] = useState<string | null>(null);
  const [navigation, setNavigation] = useState<Navigation | null>(null),
    [taskId, setTaskId] = useState<string | null>(null);
  const navigationGeneration = useRef(0);
  function changeView(next: string) {
    navigationGeneration.current += 1;
    setView(next);
  }
  function resetSelection() {
    navigationGeneration.current += 1;
    setNavigation(null);
    setTaskId(null);
    setView("overview");
    setError(null);
  }
  async function openDataset(dataset: CatalogItem) {
    resetSelection();
    try {
      await client.request("POST /api/v1/registry/open", {
        dataset_id: dataset.dataset_id,
        version: dataset.version,
      });
      setMode("dataset");
      setSessionGeneration((generation) => generation + 1);
    } catch (error) {
      setError(errorMessage(error));
    }
  }
  async function returnToRegistry() {
    resetSelection();
    try {
      await client.request("POST /api/v1/registry/close", {});
      setMode("registry");
      setSessionGeneration((generation) => generation + 1);
    } catch (error) {
      setError(errorMessage(error));
    }
  }
  async function navigate(target: Target, task?: Task) {
    const generation = ++navigationGeneration.current;
    try {
      const signalId = target.signal_id || target.signal_ids?.[0];
      let recordId = target.record_id;
      if (!recordId && signalId)
        recordId = (
          await client.request("POST /api/v1/owners/record", {
            object_type: "Signal",
            object_id: signalId,
          })
        ).record_id;
      recordId ||= task?.input_record_ids[0];
      if (generation !== navigationGeneration.current) return;
      if (!recordId) throw new Error("This target has no navigable record.");
      setError(null);
      setView("explorer");
      setNavigation({
        ...target,
        record_id: recordId,
        signal_id: signalId,
        returnTask: task?.task_id,
        sequence: generation,
      });
    } catch (error) {
      if (generation === navigationGeneration.current)
        setError(errorMessage(error));
    }
  }
  async function openAnnotation(item: Annotation) {
    if (item.object_type === "Dataset") return changeView("overview");
    if (item.object_type === "Task") {
      setTaskId(item.object_id);
      return changeView("tasks");
    }
    if (item.object_type === "Source") {
      const generation = ++navigationGeneration.current;
      try {
        const owner = await client.request("POST /api/v1/owners/record", {
          object_type: "Source",
          object_id: item.object_id,
        });
        if (generation === navigationGeneration.current)
          await navigate({
            record_id: owner.record_id,
            source_id: item.object_id,
            kind: item.span_type === "static" ? undefined : "time_interval",
            start: item.start_us,
            end: item.end_us,
          });
      } catch (error) {
        if (generation === navigationGeneration.current)
          setError(errorMessage(error));
      }
      return;
    }
    return navigate({
      ...(item.object_type === "Record"
        ? { record_id: item.object_id }
        : { signal_id: item.object_id }),
      kind: item.span_type === "static" ? "signal" : "time_interval",
      start: item.start_us,
      end: item.end_us,
    });
  }
  const status = !token
    ? "Open the authenticated launch URL printed by timenet view."
    : error ||
      session.error ||
      overview.error ||
      catalog.error ||
      connection.message;
  if (activeMode === "registry")
    return (
      <>
        <header>
          <h1>TimeNet viewer</h1>
          <p id="status" className={connection.kind} role="status">
            {status}
          </p>
          <p id="registry">Registry: {session.data?.registry}</p>
        </header>
        <Catalog
          filters={catalogFilters}
          setFilters={(filters) => {
            setCatalogFilters(filters);
            setCatalogAfter("");
          }}
          facets={
            catalog.data?.facets || { domains: [], tasks: [], licenses: [] }
          }
          total={catalog.data?.total_matches || 0}
          issues={catalog.data?.issues || []}
          next={catalog.data?.next_cursor}
          onNext={setCatalogAfter}
          onFirst={() => setCatalogAfter("")}
          datasets={catalog.data?.items || []}
          registry={session.data?.registry}
          openDataset={openDataset}
        />
      </>
    );
  return (
    <>
      <header>
        <h1>TimeNet viewer</h1>
        <p id="status" className={connection.kind} role="status">
          {status}
        </p>
        {session.data && <p id="registry">Registry: {session.data.registry}</p>}
        <nav className="tabs">
          {activeMode === "dataset" && (
            <button onClick={returnToRegistry}>Datasets</button>
          )}
          {[
            ["overview", "Overview"],
            ["explorer", "Records"],
            ["tasks", "Tasks"],
            ["annotations", "Annotations"],
          ].map(([id, label]) => (
            <button
              key={id}
              data-view={id}
              className={view === id ? "active" : ""}
              onClick={() => changeView(id)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>
      {session.data && (
        <>
          {view === "overview" && (
            <main id="overview" className="layout">
              <Overview overview={overview.data} />
            </main>
          )}
          {view === "explorer" && (
            <main id="explorer">
              <Records
                {...{ client, navigation }}
                returnToTask={
                  navigation?.returnTask
                    ? () => {
                        setTaskId(navigation.returnTask!);
                        changeView("tasks");
                      }
                    : null
                }
              />
            </main>
          )}
          {view === "tasks" && (
            <main id="tasks" className="split">
              <Tasks
                {...{ client, taskId, navigate, openAnnotation }}
                selectTask={setTaskId}
                types={overview.data?.schema.task_types || []}
              />
            </main>
          )}
          {view === "annotations" && (
            <main id="annotations" className="split">
              <Annotations {...{ client, openAnnotation }} />
            </main>
          )}
        </>
      )}
    </>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
