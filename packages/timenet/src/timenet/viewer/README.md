# TimeNet viewer

A read-only browser view of a TimeNet registry, or of one dataset version. Start it with
`timenet view`. [The CLI docs](../../../../../docs/cli/timenet.md) explain the options.

## Files

- `launch.py` opens the registry, binds `127.0.0.1` (also behind an HTTPS reverse proxy, with
  `--public-origin`), makes a new access token for each launch, and opens the browser.
- `security.py` checks every request: the Host, the Origin, and the token. It also adds browser
  security headers.
- `errors.py` holds the structured error every API route returns, the mapping of registry and
  storage failures to safe responses, and dataset reference validation.
- `tasks.py` serves the task list and one task's detail; `models.py` holds their response
  models and the projections from TimeF objects.
- `records.py` serves one record's structure with bounded hierarchy, annotation, and response
  sizes, leaving signal values lazy, and bounded windows of one scalar signal's values.
- `app.py` builds the application: the catalog and manifest API, the page, and its assets.
- `static/` holds the built frontend. It is committed, so installing `timenet[viewer]` needs no
  Node.js.

## Frontend

The source is in `frontend/` at the repository root. It uses React, TypeScript, and Vite, which
build one small script and style sheet. After you change the source, rebuild and commit `static/`:

```bash
cd frontend
npm ci
npm run build
```

`make frontend-check` rebuilds and fails if `static/` no longer matches the source. Run it before
you commit frontend changes.

`npm test` runs the frontend tests in Node.js with jsdom; no browser is needed. Each test file
calls `tests/harness.mjs` to bundle the sources it renders.

## Loading costs

Task pages limit the response, but later pages repeat a scan and hydrate skipped tasks. Opening a
task scans from the start; a missing ID scans every task. Without a split filter, the reader
hydrates all records before yielding tasks. Split filtering can still hydrate referenced records
and dependencies. Opening a remote version may download the full dataset. Signal values remain
lazy.

Opening a record hydrates the entire record hierarchy and annotations; signal windows repeat that
hydration. Selecting a signal loads its first window; values and irregular offsets are read only
when a window is requested. Windows return at most `RecordLimits.window` steps (2,000 by
default), but storage may read entire intersecting chunks.

## Remote access

`--public-origin` serves the viewer through an HTTPS reverse proxy. The viewer still binds only
`127.0.0.1`, so the token never crosses the network in plain HTTP: run the proxy on the same
machine, or reach the port through a tunnel (`ssh -L`). The origin is normalized as browsers do
(lowercase host, no default port, IDNA) and becomes the only accepted Host and Origin, and the
viewer sends `Strict-Transport-Security` for it. The proxy must forward the original `Host`
header (for nginx, `proxy_set_header Host $host;`).

Anyone who reaches the proxy with the printed URL can read every dataset in the registry, so put
the proxy behind its own authentication.

## TODO

- **Proxy on another machine.** The viewer has no non-loopback bind. Allowing one safely needs
  TLS on the viewer itself (`--ssl-certfile` and `--ssl-keyfile`, required for that bind), an
  optional `--allow-from <proxy address>` checked in the request boundary, and `ipaddress`-based
  loopback detection that also accepts IPv6.
- **Catalog scale.** Each catalog request lists the whole registry, then reads the manifests on
  one page. Large registries will need filtering and paging in the registry itself.
- **Catalog errors.** A dataset that is listed but whose manifest cannot be read makes the whole
  catalog request fail (with the mapped registry error). Report it per dataset instead.
- **Catalog links.** Filters and the page number live only in browser state. Put them in the URL
  so a catalog view can be reloaded and shared.
