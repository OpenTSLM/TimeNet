export function Overview({ overview }) {
  if (!overview) return <section className="panel">Loading manifest…</section>;
  const specs = overview.schema.tensor_specs.map((spec) => `${spec.spec_type}: [${spec.value_shape.join(", ")}]${spec.dimension_names.length ? ` (${spec.dimension_names.join(", ")})` : ""}`);
  const rows = [
    ["License", overview.identity.license], ["Access", overview.identity.access],
    ["Domains", overview.identity.domains.join(", ") || "—"], ["Tags", overview.identity.tags.join(", ") || "—"],
    ["Values backend", overview.format.values_backend], ["TimeF version", overview.format.timef_version],
    ["Counts", Object.entries(overview.counts).map(([key, value]) => `${key}: ${value}`).join("; ")],
    ["Signal specs", overview.schema.spec_types.join(", ") || "—"], ["Task types", overview.schema.task_types.join(", ") || "—"],
    ["Tensor-valued signals", specs.join("; ") || "none"],
    ["Dependencies", overview.dependencies.join(", ") || "none"], ["Source", overview.identity.source_url || "—"],
  ];
  return <section className="panel"><h2>Dataset</h2><div id="overview-selection" className="muted">Selected: Dataset overview</div><div id="overview-details">
    <h3>{overview.identity.name} ({overview.identity.dataset_id}@{overview.identity.version})</h3><p>{overview.identity.description}</p>
    <dl className="facts">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
  </div></section>;
}
