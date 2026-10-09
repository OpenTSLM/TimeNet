import { useEffect, useRef, useState } from "react";

/** Plotly owns only this component's empty host; React owns its surrounding UI. */
export function SignalPlot({ windows, annotations, clock, utc, onViewport }) {
  const host = useRef(null);
  const queue = useRef(Promise.resolve());
  const [error, setError] = useState(null);
  const handlers = useRef({ onViewport });
  handlers.current = { onViewport };
  useEffect(() => {
    const node = host.current;
    let active = true;
    let timer;
    const origin = BigInt(windows.find((data) => data.axis_kind !== "ordinal")?.items[0]?.x || 0);
    const colors = ["#1677b9", "#b54708", "#087443", "#7a5af8", "#d92d20"];
    const series = windows.map((data, index) => ({
      color: colors[index % colors.length], name: plotText(`${data.signal_name} — ${data.spec_name}`),
      unit: data.unit || "", axisKind: data.axis_kind, reduced: data.reduced,
      values: data.items.map((item) => [data.axis_kind === "ordinal" ? Number(item.x) : Number(BigInt(item.x) - origin),
        item.kind === "finite" ? Number(item.value) : null, item.index, item.x, item.display]),
    }));
    queue.current = queue.current.catch(() => {}).then(async () => {
      if (!active) return;
      await renderPlot(node, series, annotations, origin, clock, utc);
      if (!active) return;
      node.on("plotly_relayout", (event) => {
        const key = Object.keys(event).find((key) => /^xaxis\d*\.range\[0\]$/.test(key));
        if (!key) return;
        const axis = key.split(".")[0], index = Number(axis.slice(5) || "1") - 1;
        const start = event[key], stop = event[`${axis}.range[1]`];
        if (!Number.isFinite(start) || !Number.isFinite(stop)) return;
        clearTimeout(timer);
        timer = setTimeout(() => { if (active) handlers.current.onViewport(start, stop, series[index]?.axisKind, origin); }, 250);
      });
      node.on("plotly_clickannotation", (event) => {
        if (active) setError(`Selected annotation: ${event.annotation.meta?.name || event.annotation.text}`);
      });
    }).catch((error) => { if (active) setError(error.message); });
    return () => {
      active = false; clearTimeout(timer);
      node.removeAllListeners?.("plotly_relayout");
      node.removeAllListeners?.("plotly_clickannotation");
    };
  }, [windows, annotations, clock, utc]);
  useEffect(() => {
    const node = host.current;
    return () => { queue.current.finally(() => Plotly.purge(node)); };
  }, []);
  return <><div id="plot" ref={host} />{error && <p role="status">{error}</p>}</>;
}

function renderPlot(plot, plottedSeries, plotAnnotations, plotOrigin, recordClockOrigin, utc) {
  const traces = plottedSeries.map((series, index) => ({
    type: "scatter",
    mode: series.reduced || series.axisKind === "irregular" ? "markers" : "lines",
    connectgaps: false,
    xaxis: index ? `x${index + 1}` : "x",
    yaxis: index ? `y${index + 1}` : "y",
    name: series.name,
    x: series.values.map((item) => item[0]),
    y: series.values.map((item) => item[1]),
    customdata: series.values.map((item) => [...item.slice(2), recordClockOrigin !== null && series.axisKind !== "ordinal"
      ? formatUtc(recordClockOrigin + BigInt(item[3])) : "unknown"]),
    line: { color: series.color, width: 1.5 },
    hovertemplate: "%{fullData.name}<br>step=%{customdata[0]}<br>x=%{customdata[1]}<br>value=%{customdata[2]}<br>UTC=%{customdata[3]}<extra></extra>",
  }));
  const shapes = plotAnnotations.flatMap((item) => {
    if (item.span_type === "static" || item.start_us === null) return [];
    if (item.span_type === "interval") return [{
      type: "rect", xref: "x", yref: "paper", x0: Number(BigInt(item.start_us) - plotOrigin), x1: Number(BigInt(item.end_us) - plotOrigin),
      y0: 0, y1: 1, fillcolor: "rgba(217,45,32,.16)", line: { width: 0 },
    }];
    return [{ type: "line", xref: "x", yref: "paper", x0: Number(BigInt(item.start_us) - plotOrigin), x1: Number(BigInt(item.start_us) - plotOrigin), y0: 0, y1: 1, line: { color: "#d92d20", width: 1.5 } }];
  });
  const annotations = plotAnnotations.flatMap((item) => {
    if (item.span_type === "static" || item.start_us === null) return [];
    const end = item.end_us === null ? item.start_us : item.end_us;
    return [{ x: (Number(BigInt(item.start_us) - plotOrigin) + Number(BigInt(end) - plotOrigin)) / 2, y: 1, yref: "paper", text: plotText(item.name),
      showarrow: false, yanchor: "bottom", font: { color: "#b42318", size: 11 }, captureevents: true,
      meta: item }];
  });
  const axes = {};
  const firstTemporal = plottedSeries.findIndex((series) => series.axisKind !== "ordinal");
  plottedSeries.forEach((series, index) => {
    const suffix = index ? String(index + 1) : "";
    axes[`xaxis${suffix}`] = {
      title: series.axisKind === "ordinal" ? "Step" : `Time offset from ${plotOrigin} µs`,
      matches: series.axisKind !== "ordinal" && index !== firstTemporal
        ? (firstTemporal ? `x${firstTemporal + 1}` : "x") : undefined,
    };
    axes[`yaxis${suffix}`] = { title: series.unit || "Value" };
    if (utc && recordClockOrigin !== null && series.axisKind !== "ordinal" && series.values.length) {
      const start = series.values[0][0], stop = series.values.at(-1)[0];
      const ticks = Array.from({ length: 5 }, (_, i) => start + (stop - start) * i / 4);
      Object.assign(axes[`xaxis${suffix}`], { title: "UTC", tickmode: "array", tickvals: ticks,
        ticktext: ticks.map((value) => formatUtc(recordClockOrigin + plotOrigin + BigInt(Math.round(value)))) });
    }
  });
  return Plotly.react(plot, traces, {
    margin: { l: 70, r: 24, t: 24, b: 54 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    ...axes,
    grid: { rows: Math.max(1, traces.length), columns: 1, pattern: "independent" },
    height: Math.max(360, traces.length * 220),
    hovermode: "x unified", showlegend: traces.length > 1, shapes, annotations,
  }, { displaylogo: false, responsive: true, scrollZoom: true, modeBarButtonsToRemove: ["toImage", "sendDataToCloud"] })
;
}

function formatUtc(microseconds) {
  const seconds = microseconds >= 0n ? microseconds / 1000000n : (microseconds - 999999n) / 1000000n;
  const fraction = microseconds - seconds * 1000000n;
  const date = new Date(Number(seconds * 1000n));
  if (!Number.isFinite(date.getTime())) return String(microseconds) + " unix µs";
  return date.toISOString().replace(/\.\d{3}Z$/, `.${String(fraction).padStart(6, "0")}Z`);
}

function plotText(value) {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

