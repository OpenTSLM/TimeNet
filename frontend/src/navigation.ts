export type ViewerProps = {
  dataset: string;
  version: string;
  search: string;
  navigate: (params: URLSearchParams, options?: { replace?: boolean }) => void;
  getJson: <T>(path: string, signal: AbortSignal) => Promise<T>;
};

type Changes = Record<string, string | null>;

export function changeSelection(search: string, changes: Changes): URLSearchParams {
  const params = new URLSearchParams(search);
  params.delete("token");
  for (const [key, value] of Object.entries(changes)) {
    if (value === null || value === "") params.delete(key);
    else params.set(key, value);
  }
  return params;
}

export function selectRecord(search: string, id: string): URLSearchParams {
  return changeSelection(search, { record: id, source: null, signal: null, start: null, stop: null });
}

export function selectTask(search: string, id: string | null): URLSearchParams {
  return changeSelection(search, { view: "tasks", task: id, record: null, source: null, signal: null, start: null, stop: null });
}

export function inspectorHref(search: string, record: string): string {
  return `/?${changeSelection(search, {
    view: "records", record,
  })}`;
}

export function returnFromInspector(search: string): URLSearchParams {
  const params = new URLSearchParams(search);
  if (params.get("task")) return changeSelection(search, { view: "tasks" });
  return selectTask(search, null);
}

export type RecordChoice = { id: string; roles: string[] };

export function recordChoices(inputs: string[], targets: string[], candidates: string[]): RecordChoice[] {
  const choices = new Map<string, RecordChoice>();
  for (const [role, ids] of [["Input", inputs], ["Target", targets], ["Candidate", candidates]] as const) {
    for (const id of ids) {
      const choice = choices.get(id) ?? { id, roles: [] };
      if (!choice.roles.includes(role)) choice.roles.push(role);
      choices.set(id, choice);
    }
  }
  return [...choices.values()];
}
