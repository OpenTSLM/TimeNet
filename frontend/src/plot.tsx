import type * as PlotlyTypes from "plotly.js";
import type { Annotation, WindowData } from "./types";
declare const Plotly: typeof PlotlyTypes;
export type ViewportHandler = (
  first: number,
  last: number,
  kind: WindowData["axis_kind"],
  origin: bigint,
) => void;
interface PlotProps {
  windows: WindowData[];
  annotations: Annotation[];
  clock: bigint | null;
  utc: boolean;
  onViewport: ViewportHandler;
  annotationSignalId: string;
}
interface PlotSample {
  x: number;
  y: number | null;
  index: string;
  exactX: string;
  display: string | null;
}
interface PlotSeries {
  color: string;
  name: string;
  signalId: string;
  unit: string;
  axisKind: WindowData["axis_kind"];
  reduced: boolean;
  values: PlotSample[];
}
import { useEffect, useRef, useState } from "react";

/** Plotly owns only this component's empty host; React owns its surrounding UI. */
export function SignalPlot({
  windows,
  annotations,
  clock,
  utc,
  onViewport,
  annotationSignalId,
}: PlotProps) {
  const host = useRef<(HTMLDivElement & PlotlyTypes.PlotlyHTMLElement) | null>(
    null,
  );
  const queue = useRef(Promise.resolve());
  const [error, setError] = useState<string | null>(null);
  const handlers = useRef({ onViewport });
  handlers.current = { onViewport };
  useEffect(() => {
    const node = host.current;
    if (!node) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const origin = BigInt(
      windows.find((data) => data.axis_kind !== "ordinal")?.items[0]?.x || 0,
    );
    const colors = ["#1677b9", "#b54708", "#087443", "#7a5af8", "#d92d20"];
    const series: PlotSeries[] = windows.map((data, index) => ({
      color: colors[index % colors.length],
      name: plotText(`${data.signal_name} — ${data.spec_name}`),
      signalId: data.signal_id,
      unit: data.unit || "",
      axisKind: data.axis_kind,
      reduced: data.reduced,
      values: data.items.map((item) => ({
        x:
          data.axis_kind === "ordinal"
            ? Number(item.x)
            : Number(BigInt(item.x) - origin),
        y: item.kind === "finite" ? Number(item.value) : null,
        index: item.index,
        exactX: item.x,
        display: item.display,
      })),
    }));
    queue.current = queue.current
      .catch(() => {})
      .then(async () => {
        if (!active) return;
        await renderPlot(
          node,
          series,
          annotations,
          origin,
          clock,
          utc,
          annotationSignalId,
        );
        if (!active) return;
        node.on("plotly_relayout", (event) => {
          const key = Object.keys(event).find((key) =>
            /^xaxis\d*\.range\[0\]$/.test(key),
          );
          if (!key) return;
          const axis = key.split(".")[0],
            index = Number(axis.slice(5) || "1") - 1;
          const values: Record<string, unknown> = { ...event };
          const start = values[key],
            stop = values[`${axis}.range[1]`];
          if (
            typeof start !== "number" ||
            typeof stop !== "number" ||
            !Number.isFinite(start) ||
            !Number.isFinite(stop)
          )
            return;
          clearTimeout(timer);
          timer = setTimeout(() => {
            if (active)
              handlers.current.onViewport(
                start,
                stop,
                series[index]?.axisKind || "regular",
                origin,
              );
          }, 250);
        });
        node.on("plotly_clickannotation", (event) => {
          if (active) setError(`Selected annotation: ${event.annotation.text}`);
        });
      })
      .catch((error) => {
        if (active) setError(error.message);
      });
    return () => {
      active = false;
      clearTimeout(timer);
      node.removeAllListeners?.("plotly_relayout");
      node.removeAllListeners?.("plotly_clickannotation");
    };
  }, [windows, annotations, clock, utc, annotationSignalId]);
  useEffect(() => {
    const node = host.current;
    return () => {
      queue.current.finally(() => {
        if (node) Plotly.purge(node);
      });
    };
  }, []);
  return (
    <>
      <div id="plot" ref={host} />
      {error && <p role="status">{error}</p>}
    </>
  );
}

function renderPlot(
  plot: PlotlyTypes.PlotlyHTMLElement,
  plottedSeries: PlotSeries[],
  plotAnnotations: Annotation[],
  plotOrigin: bigint,
  recordClockOrigin: bigint | null,
  utc: boolean,
  annotationSignalId: string,
) {
  const traces = buildTraces(plottedSeries, recordClockOrigin);
  const { shapes, annotations } = buildOverlays(
    plottedSeries,
    plotAnnotations,
    plotOrigin,
    annotationSignalId,
  );
  const axes = buildAxes(plottedSeries, plotOrigin, recordClockOrigin, utc);
  return Plotly.react(
    plot,
    traces,
    {
      margin: { l: 70, r: 24, t: 24, b: 54 },
      paper_bgcolor: "#fff",
      plot_bgcolor: "#fff",
      ...axes,
      grid: {
        rows: Math.max(1, traces.length),
        columns: 1,
        pattern: "independent",
      },
      height: Math.max(360, traces.length * 220),
      hovermode: "x unified",
      showlegend: traces.length > 1,
      shapes,
      annotations,
    },
    {
      displaylogo: false,
      responsive: true,
      scrollZoom: true,
      modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
    },
  );
}

function formatUtc(microseconds: bigint) {
  const seconds =
    microseconds >= 0n
      ? microseconds / 1000000n
      : (microseconds - 999999n) / 1000000n;
  const fraction = microseconds - seconds * 1000000n;
  const date = new Date(Number(seconds * 1000n));
  if (!Number.isFinite(date.getTime()))
    return String(microseconds) + " unix µs";
  return date
    .toISOString()
    .replace(/\.\d{3}Z$/, `.${String(fraction).padStart(6, "0")}Z`);
}

function plotText(value: string) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function buildTraces(
  plottedSeries: PlotSeries[],
  recordClockOrigin: bigint | null,
): PlotlyTypes.Data[] {
  const traces: PlotlyTypes.Data[] = plottedSeries.map((series, index) => ({
    type: "scatter",
    mode:
      series.reduced || series.axisKind === "irregular" ? "markers" : "lines",
    connectgaps: false,
    xaxis: index ? `x${index + 1}` : "x",
    yaxis: index ? `y${index + 1}` : "y",
    name: series.name,
    x: series.values.map((item) => item.x),
    y: series.values.map((item) => item.y),
    customdata: series.values.map((item) => [
      item.index,
      item.exactX,
      item.display,
      recordClockOrigin !== null && series.axisKind !== "ordinal"
        ? formatUtc(recordClockOrigin + BigInt(item.exactX))
        : "unknown",
    ]),
    line: { color: series.color, width: 1.5 },
    hovertemplate:
      "%{fullData.name}<br>step=%{customdata[0]}<br>x=%{customdata[1]}<br>value=%{customdata[2]}<br>UTC=%{customdata[3]}<extra></extra>",
  }));
  return traces;
}

function buildOverlays(
  plottedSeries: PlotSeries[],
  plotAnnotations: Annotation[],
  plotOrigin: bigint,
  annotationSignalId: string,
) {
  // Overlays belong to the signal queried by WindowPanel, not the first subplot.
  const annotationIndex = plottedSeries.findIndex(
    (series) =>
      series.signalId === annotationSignalId && series.axisKind !== "ordinal",
  );
  const temporalAnnotations = annotationIndex < 0 ? [] : plotAnnotations;
  const xref = (
    annotationIndex ? `x${annotationIndex + 1}` : "x"
  ) as PlotlyTypes.Shape["xref"];
  const yref = (
    annotationIndex ? `y${annotationIndex + 1} domain` : "y domain"
  ) as PlotlyTypes.Shape["yref"];
  const shapes = temporalAnnotations.flatMap<Partial<PlotlyTypes.Shape>>(
    (item) => {
      if (item.span_type === "static" || item.start_us === null) return [];
      if (item.span_type === "interval")
        return [
          {
            type: "rect",
            xref,
            yref,
            x0: Number(BigInt(item.start_us) - plotOrigin),
            x1: Number(BigInt(item.end_us ?? item.start_us) - plotOrigin),
            y0: 0,
            y1: 1,
            fillcolor: "rgba(217,45,32,.16)",
            line: { width: 0 },
          },
        ];
      return [
        {
          type: "line",
          xref,
          yref,
          x0: Number(BigInt(item.start_us) - plotOrigin),
          x1: Number(BigInt(item.start_us) - plotOrigin),
          y0: 0,
          y1: 1,
          line: { color: "#d92d20", width: 1.5 },
        },
      ];
    },
  );
  const annotations = temporalAnnotations.flatMap<
    Partial<PlotlyTypes.Annotations>
  >((item) => {
    if (item.span_type === "static" || item.start_us === null) return [];
    const end = item.end_us === null ? item.start_us : item.end_us;
    return [
      {
        x:
          (Number(BigInt(item.start_us) - plotOrigin) +
            Number(BigInt(end) - plotOrigin)) /
          2,
        y: 1,
        xref,
        yref,
        text: plotText(item.name),
        showarrow: false,
        yanchor: "bottom",
        font: { color: "#b42318", size: 11 },
        captureevents: true,
        meta: item,
      },
    ];
  });
  return { shapes, annotations };
}

function buildAxes(
  plottedSeries: PlotSeries[],
  plotOrigin: bigint,
  recordClockOrigin: bigint | null,
  utc: boolean,
): Record<string, Partial<PlotlyTypes.LayoutAxis>> {
  const axes: Record<string, Partial<PlotlyTypes.LayoutAxis>> = {};
  const firstTemporal = plottedSeries.findIndex(
    (series) => series.axisKind !== "ordinal",
  );
  plottedSeries.forEach((series, index) => {
    const suffix = index ? String(index + 1) : "";
    axes[`xaxis${suffix}`] = {
      title: {
        text:
          series.axisKind === "ordinal"
            ? "Step"
            : `Time offset from ${plotOrigin} µs`,
      },
      matches:
        series.axisKind !== "ordinal" && index !== firstTemporal
          ? ((firstTemporal
              ? `x${firstTemporal + 1}`
              : "x") as PlotlyTypes.AxisName)
          : undefined,
    };
    axes[`yaxis${suffix}`] = { title: { text: series.unit || "Value" } };
    if (
      utc &&
      recordClockOrigin !== null &&
      series.axisKind !== "ordinal" &&
      series.values.length
    ) {
      const start = series.values[0].x,
        stop = series.values.at(-1)!.x;
      const ticks = Array.from(
        { length: 5 },
        (_, i) => start + ((stop - start) * i) / 4,
      );
      Object.assign(axes[`xaxis${suffix}`], {
        title: { text: "UTC" },
        tickmode: "array",
        tickvals: ticks,
        ticktext: ticks.map((value) =>
          formatUtc(recordClockOrigin + plotOrigin + BigInt(Math.round(value))),
        ),
      });
    }
  });
  return axes;
}
