# TimeNet viewer

A read-only browser view of a TimeNet registry, or of one dataset version.

## Files

- `security.py` checks every request: the Host, the Origin, and the token. It also adds browser
  security headers.
- `errors.py` holds the structured error every API route returns, the mapping of registry and
  storage failures to safe responses, and dataset reference validation.
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

## TODO

- **Catalog scale.** Each catalog request lists the whole registry, then reads the manifests on
  one page. Large registries will need filtering and paging in the registry itself.
- **Catalog errors.** A dataset that is listed but whose manifest cannot be read makes the whole
  catalog request fail (with the mapped registry error). Report it per dataset instead.
- **Catalog links.** Filters and the page number live only in browser state. Put them in the URL
  so a catalog view can be reloaded and shared.
