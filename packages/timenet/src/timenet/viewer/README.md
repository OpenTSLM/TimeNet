# TimeNet viewer

A read-only browser view of a TimeNet registry, or of one dataset version. Start it with
`timenet view`. [The CLI docs](../../../../../docs/cli/timenet.md) explain the options.

## Files

- `launch.py` opens the registry, binds `127.0.0.1` (or, with `--public-origin`, a host behind an
  HTTPS reverse proxy), makes a new access token for each launch, and opens the browser.
- `security.py` checks every request: the Host, the Origin, and the token. It also adds browser
  security headers.
- `app.py` serves the catalog and manifest API, the page, and its assets.
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

## Remote access

`--public-origin` serves the viewer through an HTTPS reverse proxy. `--host` allows a
non-loopback IPv4 bind, and only together with `--public-origin`. The origin is normalized as
browsers do (lowercase host, no default port, IDNA) and becomes the only accepted Host and Origin.
The proxy must forward the original `Host` header (for nginx, `proxy_set_header Host $host;`).

Anyone who reaches the proxy with the printed URL can read every dataset in the registry, so put
the proxy behind its own authentication.

## TODO

- **Catalog scale.** Each catalog request lists the whole registry, then reads the manifests on
  one page. Large registries will need filtering and paging in the registry itself.
- **Catalog errors.** A dataset that is listed but whose manifest cannot be read makes the whole
  catalog request fail with a 500. Report it per dataset instead.
- **Catalog links.** Filters and the page number live only in browser state. Put them in the URL
  so a catalog view can be reloaded and shared.
