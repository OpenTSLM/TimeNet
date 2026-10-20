# TimeNet viewer

A read-only browser view of a TimeNet registry, or of one dataset version. Start it with
`timenet view`. [The CLI docs](../../../../../docs/cli/timenet.md) explain the options.

## Files

- `launch.py` opens the registry, binds `127.0.0.1`, makes a new access token for each launch,
  and opens the browser.
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

## TODO

- **Remote access through an HTTPS reverse proxy.** v0 binds only to `127.0.0.1`. An earlier
  revision of PR #470 had remote access. Adding it back needs:
  - `--host` for a non-loopback bind, accepted only with `--public-origin`.
  - `--public-origin`, checked to be HTTPS and normalized as browsers do (lowercase host, no
    default port, IDNA). It becomes the only accepted Host and Origin, and the base of the
    printed URL.
  - Docs that say the proxy must forward the original `Host` header (for nginx,
    `proxy_set_header Host $host;`), and a threat model for using the token beyond loopback.
  - Tests for origin normalization, port mismatches, and binds that must fail.
- **Catalog scale.** Each catalog request lists the whole registry, then reads the manifests on
  one page. Large registries will need filtering and paging in the registry itself.
- **Catalog errors.** A dataset that is listed but whose manifest cannot be read makes the whole
  catalog request fail with a 500. Report it per dataset instead.
- **Catalog links.** Filters and the page number live only in browser state. Put them in the URL
  so a catalog view can be reloaded and shared.
