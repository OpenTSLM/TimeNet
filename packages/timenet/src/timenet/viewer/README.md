# TimeNet viewer

A read-only browser viewer for a TimeNet registry, or for one pinned dataset version. Launch it
with `timenet view`; [the CLI docs](../../../../../docs/cli/timenet.md) cover usage.

## Layout

- `launch.py` binds `127.0.0.1` (or, with `--public-origin`, a host behind an HTTPS reverse
  proxy), creates the per-launch token, and opens the browser.
- `security.py` is the request boundary: browser-normalized origins, Host and Origin checks, the
  bearer token, and browser security headers.
- `app.py` serves the catalog and manifest API, plus the packaged shell and assets.
- `static/` holds the frontend build. It is committed so the package works without Node. The
  source lives in `frontend/` at the repository root; rebuild it after changing the source:

```bash
cd frontend
npm ci
npm run build
```

## Remote access

`--public-origin` serves the viewer through an authenticated HTTPS reverse proxy, and `--host`
allows a non-loopback IPv4 bind only together with it. The origin is normalized as browsers do
(lowercase host, default port dropped, IDNA) and becomes the only accepted Host and Origin. The
proxy must forward the original `Host` header (for nginx, `proxy_set_header Host $host;`).

Anyone who reaches the proxy with the printed URL can read every dataset in the registry, so put
the proxy behind its own authentication.
