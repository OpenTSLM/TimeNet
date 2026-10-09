import { createRoot } from "react-dom/client";
import { useRef, useState } from "react";
import { ApiClient } from "./api";
import { useResource } from "./hooks";
import { Overview } from "./overview";
import { Records } from "./records";
import { Tasks } from "./tasks";
import { Annotations } from "./annotations";

const token = new URLSearchParams(location.hash.slice(1)).get("token");
history.replaceState(null, "", location.pathname);
const client = new ApiClient(token || "");

function App() {
  const session = useResource(client, token ? "/api/v1/session" : null);
  const overview = useResource(client, session.data ? "/api/v1/overview" : null);
  const [view, setView] = useState("overview"), [error, setError] = useState(null);
  const [navigation, setNavigation] = useState(null), [taskId, setTaskId] = useState(null);
  const navigationGeneration = useRef(0);
  function changeView(next) { navigationGeneration.current += 1; setView(next); }
  async function navigate(target, task) {
    const generation = ++navigationGeneration.current;
    try {
      const signalId = target.signal_id || target.signal_ids?.[0];
      let recordId = target.record_id;
      if (!recordId && signalId) recordId = (await client.request("/api/v1/owners/record", { object_type: "Signal", object_id: signalId })).record_id;
      recordId ||= task?.input_record_ids[0];
      if (generation !== navigationGeneration.current) return;
      if (!recordId) throw new Error("This target has no navigable record.");
      setError(null); setView("explorer");
      setNavigation({ ...target, record_id: recordId, signal_id: signalId, returnTask: task?.task_id, sequence: generation });
    } catch (error) { if (generation === navigationGeneration.current) setError(error.message); }
  }
  async function openAnnotation(item) {
    if (item.object_type === "Dataset") return changeView("overview");
    if (item.object_type === "Task") { setTaskId(item.object_id); return changeView("tasks"); }
    if (item.object_type === "Source") {
      const generation = ++navigationGeneration.current;
      try {
        const owner = await client.request("/api/v1/owners/record", { object_type: "Source", object_id: item.object_id });
        if (generation === navigationGeneration.current) await navigate({ record_id: owner.record_id });
      } catch (error) { if (generation === navigationGeneration.current) setError(error.message); }
      return;
    }
    return navigate({ ...(item.object_type === "Record" ? { record_id: item.object_id } : { signal_id: item.object_id }),
      kind: item.span_type === "static" ? "signal" : "time_interval", start: item.start_us, end: item.end_us });
  }
  const status = !token ? "Open the authenticated launch URL printed by timenet view." : error || session.error || overview.error || (session.data ? "Connected" : "Authenticating local session…");
  return <><header><h1>TimeNet viewer</h1><p id="status" role="status">{status}</p>{session.data && <p id="registry">Registry: {session.data.registry}</p>}<nav className="tabs">
    {[["overview", "Overview"], ["explorer", "Records"], ["tasks", "Tasks"], ["annotations", "Annotations"]].map(([id, label]) =>
      <button key={id} data-view={id} className={view === id ? "active" : ""} onClick={() => changeView(id)}>{label}</button>)}
  </nav></header>{session.data && <>
    {view === "overview" && <main id="overview" className="layout"><Overview overview={overview.data} /></main>}
    {view === "explorer" && <main id="explorer"><Records {...{ client, navigation }}
      returnToTask={navigation?.returnTask ? () => { setTaskId(navigation.returnTask); changeView("tasks"); } : null} /></main>}
    {view === "tasks" && <main id="tasks" className="split"><Tasks {...{ client, taskId, navigate, openAnnotation }}
      selectTask={setTaskId} types={overview.data?.schema.task_types || []} /></main>}
    {view === "annotations" && <main id="annotations" className="split"><Annotations {...{ client, openAnnotation }} /></main>}
  </>}</>;
}

createRoot(document.getElementById("root")).render(<App />);
