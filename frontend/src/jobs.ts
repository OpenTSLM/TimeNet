import { ApiClient, type Job } from "./api";

function pause(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(new DOMException("Cancelled", "AbortError")); };
    const timer = setTimeout(() => { signal.removeEventListener("abort", abort); resolve(); }, 100);
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
  });
}

/** A request lifetime owns both polling and any server job admitted after disposal. */
export class JobRun<T> {
  private readonly controller = new AbortController();
  private jobId: string | undefined;
  private disposed = false;
  private terminal = false;
  private cancelRequested = false;

  constructor(private readonly client: ApiClient) {}

  async start(path: string, body: unknown, progress: (job: Job) => void = () => {}): Promise<T> {
    // Admission is deliberately not aborted: its returned ID must be cancelled after disposal.
    const admitted = await this.client.request<Job>(path, body);
    this.jobId = admitted.job_id;
    if (this.disposed) {
      await this.client.cancel(this.jobId).catch(() => {});
      throw new DOMException("Cancelled", "AbortError");
    }
    if (this.cancelRequested) await this.client.cancel(this.jobId);
    while (true) {
      const job = await this.client.request<Job>(`/api/v1/jobs/${this.jobId}`, undefined, this.controller.signal);
      progress(job);
      if (["succeeded", "cancelled", "failed"].includes(job.state)) {
        this.terminal = true;
        if (job.state === "failed") throw new Error(job.error || "operation failed");
        return this.client.request<T>(`/api/v1/jobs/${this.jobId}/result`, undefined, this.controller.signal);
      }
      await pause(this.controller.signal);
    }
  }

  cancel(): Promise<unknown> {
    this.cancelRequested = true;
    return this.jobId ? this.client.cancel(this.jobId) : Promise.resolve();
  }

  dispose(): void {
    this.disposed = true;
    this.controller.abort();
    if (this.jobId && !this.terminal) void this.client.cancel(this.jobId).catch(() => {});
  }
}
