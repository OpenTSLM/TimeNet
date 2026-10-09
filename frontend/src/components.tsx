import { useState } from "react";

export function ComponentPicker({ shape, onChoose, onCancel }: {
  shape: number[]; onChoose: (indexes: number[]) => void; onCancel: () => void;
}) {
  const [indexes, setIndexes] = useState(shape.map(() => 0));
  const valid = indexes.every((value, i) => Number.isInteger(value) && value >= 0 && value < shape[i]);
  return <form onSubmit={(event) => { event.preventDefault(); if (valid) onChoose(indexes); }}>
    <p>Select one component from shape [{shape.join(", ")}].</p>
    {shape.map((size, i) => <label key={i}>Dimension {i + 1}
      <input aria-label={`Dimension ${i + 1}`} type="number" min={0} max={size - 1} value={indexes[i]}
        onChange={(event) => setIndexes(indexes.map((value, j) => j === i ? Number(event.target.value) : value))} />
    </label>)}
    <button type="submit" disabled={!valid}>Inspect component</button>
    <button type="button" onClick={onCancel}>Cancel</button>
  </form>;
}

export function Pagination({ paging, next }: {
  paging: { page: number; previous: () => void; next: (cursor: string) => void }; next?: string | null;
}) {
  return <div className="pagination">
    <button disabled={paging.page === 0} onClick={paging.previous}>Previous</button>
    <span className="muted">Page {paging.page + 1} · 10 per page</span>
    <button disabled={!next} onClick={() => next && paging.next(next)}>Next</button>
  </div>;
}
