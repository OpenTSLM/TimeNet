import type { components } from "./generated/api";

import {
  endpoints,
  type RequestBodies,
  type ResponseBodies,
  type EndpointOptions,
} from "./generated/endpoints";
export type Endpoint = keyof RequestBodies;
export type { RequestBodies, ResponseBodies };
export type Schemas = components["schemas"];
export type WindowQuery = Schemas["WindowQuery"];
export interface ConnectionStatus {
  kind: "connecting" | "connected" | "offline" | "error";
  message: string;
}
export type RequestOptions<P extends Endpoint = Endpoint> = EndpointOptions[P];
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Request failed";
}

export class ApiClient {
  private status: ConnectionStatus = {
    kind: "connecting",
    message: "Connecting to local viewer…",
  };
  private readonly listeners = new Set<(status: ConnectionStatus) => void>();

  constructor(private readonly token: string) {}

  subscribe(listener: (status: ConnectionStatus) => void): () => void {
    this.listeners.add(listener);
    listener(this.status);
    return () => this.listeners.delete(listener);
  }

  async request<P extends Endpoint>(
    path: P,
    body: RequestBodies[P],
    options?: RequestOptions<P>,
  ): Promise<ResponseBodies[P]> {
    let response: Response;
    try {
      const url = new URL(endpoints[path].path, location.origin);
      for (const [key, value] of Object.entries(options?.query || {}))
        url.searchParams.set(key, String(value));
      response = await fetch(url, {
        method: endpoints[path].method,
        signal: options?.signal,
        headers: {
          Authorization: `Bearer ${this.token}`,
          ...(body === undefined ? {} : { "Content-Type": "application/json" }),
        },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        this.updateStatus({
          kind: "offline",
          message: "Connection lost. Restart the viewer and retry.",
        });
      }
      throw error;
    }
    const data = await response.json();
    if (!response.ok) {
      const message =
        data.message ||
        (Array.isArray(data.detail)
          ? data.detail
              .map(
                (item: { loc?: string[]; msg?: string }) =>
                  `${item.loc?.join(".")}: ${item.msg}`,
              )
              .join("; ")
          : data.detail) ||
        "request failed";
      this.updateStatus({
        kind: "error",
        message: `Request failed: ${message}`,
      });
      throw new Error(message);
    }
    this.updateStatus({ kind: "connected", message: "Connected" });
    return data as ResponseBodies[P];
  }

  private updateStatus(status: ConnectionStatus): void {
    this.status = status;
    this.listeners.forEach((listener) => listener(status));
  }
}
