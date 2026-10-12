import { useEffect, useState } from "react";
import {
  ApiClient,
  type ConnectionStatus,
  type Endpoint,
  type RequestBodies,
  type ResponseBodies,
  type RequestOptions,
} from "./api";

export function useConnectionStatus(client: ApiClient): ConnectionStatus {
  const [status, setStatus] = useState<ConnectionStatus>({
    kind: "connecting",
    message: "Connecting to local viewer…",
  });
  useEffect(() => client.subscribe(setStatus), [client]);
  return status;
}

export function useResource<P extends Endpoint>(
  client: ApiClient,
  path: P | null,
  body: RequestBodies[P],
  options?: RequestOptions<P>,
) {
  const key = JSON.stringify([path, body, options]);
  const [state, setState] = useState<{
    key: string;
    data?: ResponseBodies[P];
    error?: string;
  }>({ key });
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    let current = true;
    setState({ key });
    client
      .request(
        path,
        body,
        Object.assign({}, options, { signal: controller.signal }),
      )
      .then(
        (data) => {
          if (current) setState({ key, data });
        },
        (error) => {
          if (current && error.name !== "AbortError")
            setState({ key, error: error.message });
        },
      );
    return () => {
      current = false;
      controller.abort();
    };
  }, [client, key]);
  return state.key === key ? state : { key };
}

export function useCursor(key: string) {
  const [state, setState] = useState<{
    key: string;
    cursors: (string | null)[];
    index: number;
  }>({ key, cursors: [null], index: 0 });
  const current =
    state.key === key ? state : { key, cursors: [null], index: 0 };
  return {
    after: current.cursors[current.index],
    page: current.index,
    previous: () =>
      setState({ ...current, index: Math.max(0, current.index - 1) }),
    next: (cursor: string) =>
      setState({
        key,
        cursors: [...current.cursors.slice(0, current.index + 1), cursor],
        index: current.index + 1,
      }),
  };
}
