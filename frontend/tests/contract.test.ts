import type { ApiClient } from "../src/api";

// Compile-time regressions: endpoint choice must constrain both sides of the wire.
export async function checkEndpointTypes(client: ApiClient) {
  const page = await client.request("POST /api/v1/records/query", { limit: 10 });
  page.items[0].record_id.toUpperCase();
  // @ts-expect-error A record page is not a task page.
  page.items[0].task_id;
  // @ts-expect-error Windows require a signal ID.
  client.request("POST /api/v1/windows", { record_id: "record" });
  // @ts-expect-error A required body cannot be omitted altogether.
  client.request("POST /api/v1/windows");
  // @ts-expect-error Misspelled fields must not escape through an unknown body.
  client.request("POST /api/v1/records/query", { limti: 10 });
  // @ts-expect-error The response type cannot be chosen independently of the endpoint.
  client.request<{ invented: string }>("GET /api/v1/session");
  // @ts-expect-error A method cannot be overridden independently of the operation.
  client.request("GET /api/v1/session", undefined, {method: "DELETE"});
  // @ts-expect-error Query parameters are specific to the endpoint.
  client.request("GET /api/v1/registry/datasets", undefined, {query: {limti: "1"}});
}
