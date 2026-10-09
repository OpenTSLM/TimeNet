import type { components } from "./generated/api";

type Schemas = components["schemas"];
export type WindowQuery = Schemas["WindowQuery"];
export interface Progress { completed?: string; total?: string | null; phase?: string; unit?: string }
export interface Job { job_id: string; state: string; error?: string; progress?: Progress }
export type RequestBodies = {
  "/api/v1/records/query": Schemas["RecordQuery"];
  "/api/v1/records/detail": Schemas["RecordDetailQuery"];
  "/api/v1/sources/query": Schemas["SourceQuery"];
  "/api/v1/signals/query": Schemas["SignalQuery"];
  "/api/v1/windows": WindowQuery;
  "/api/v1/jobs/windows": WindowQuery;
  "/api/v1/tasks/query": Schemas["TaskQuery"];
  "/api/v1/tasks/detail": Schemas["TaskDetailQuery"];
  "/api/v1/annotations/query": Schemas["AnnotationQuery"];
  "/api/v1/annotations/window": Schemas["AnnotationWindowQuery"];
};

export class ApiClient {
  constructor(private readonly token: string) {}

  async request<T>(path: string, body?: unknown, signal?: AbortSignal, method?: string): Promise<T> {
    const response = await fetch(path, {
      method: method || (body === undefined ? "GET" : "POST"), signal,
      headers: { Authorization: `Bearer ${this.token}`, ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || "request failed");
    return data as T;
  }

  post<P extends keyof RequestBodies, T>(path: P, body: RequestBodies[P], signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, body, signal);
  }

  cancel(jobId: string): Promise<unknown> {
    return this.request(`/api/v1/jobs/${jobId}`, undefined, undefined, "DELETE");
  }
}

export function progressLabel(job: Job): string {
  const progress = job.progress || {};
  const count = progress.completed ?? "0";
  const total = progress.total == null ? "total unknown" : `of ${progress.total}`;
  return `${job.state}: ${progress.phase || "waiting"} — ${count} ${total} ${progress.unit || "items"}`;
}
