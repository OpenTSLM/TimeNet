import { useEffect, useRef, useState } from "react";
import { ApiClient, type Job } from "./api";
import { JobRun } from "./jobs";

export function useResource<T>(client: ApiClient, path: string | null, body?: unknown) {
  const key = JSON.stringify([path, body]);
  const [state, setState] = useState<{ key: string; data?: T; error?: string }>({ key });
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    let current = true;
    setState({ key });
    client.request<T>(path, body, controller.signal).then(
      (data) => { if (current) setState({ key, data }); },
      (error) => { if (current && error.name !== "AbortError") setState({ key, error: error.message }); },
    );
    return () => { current = false; controller.abort(); };
  }, [client, key]);
  return state.key === key ? state : { key };
}

export function useJob<T>(client: ApiClient, request: { path: string; body: unknown; sequence: number } | null) {
  const run = useRef<JobRun<T> | null>(null);
  const [state, setState] = useState<{ job?: Job; data?: T; error?: string }>({});
  const key = JSON.stringify(request);
  useEffect(() => {
    setState({});
    if (!request) return;
    const operation = new JobRun<T>(client);
    run.current = operation;
    let current = true;
    operation.start(request.path, request.body, (job) => { if (current) setState({ job }); }).then(
      (data) => { if (current) setState((state) => ({ ...state, data })); },
      (error) => { if (current && error.name !== "AbortError") setState((state) => ({ ...state, error: error.message })); },
    );
    return () => { current = false; operation.dispose(); };
  }, [client, key]);
  return { ...state, cancel: () => run.current?.cancel() };
}

export function useCursor(key: string) {
  const [state, setState] = useState<{ key: string; cursors: (string | null)[]; index: number }>({ key, cursors: [null], index: 0 });
  const current = state.key === key ? state : { key, cursors: [null], index: 0 };
  return {
    after: current.cursors[current.index], page: current.index,
    previous: () => setState({ ...current, index: Math.max(0, current.index - 1) }),
    next: (cursor: string) => setState({ key, cursors: [...current.cursors.slice(0, current.index + 1), cursor], index: current.index + 1 }),
  };
}
