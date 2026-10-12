import type { ApiClient, Schemas, WindowQuery } from "./api";
export type Annotation = Schemas["AnnotationItem"];
export type Task = Schemas["TaskDetail"];
export type Signal = Schemas["SignalItem"];
export type WindowData = Schemas["WindowResult"];
export type CatalogItem = Schemas["CatalogItem"];
export type Target = { source_id?: string } & Partial<
  Pick<
    Schemas["TargetItem"],
    "record_id" | "signal_id" | "signal_ids" | "kind" | "start" | "end"
  >
>;
export type Navigation = Target & {
  record_id: string;
  returnTask?: string;
  sequence: number;
};
export type Navigate = (target: Target, task?: Task) => void | Promise<void>;
export type OpenAnnotation = (annotation: Annotation) => void | Promise<void>;
export interface Selection {
  signal: Signal;
  component: number[];
}
export type WindowSelection = Pick<
  WindowQuery,
  "start" | "stop" | "start_us" | "end_us" | "full"
>;
export interface ClientProps {
  client: ApiClient;
}
