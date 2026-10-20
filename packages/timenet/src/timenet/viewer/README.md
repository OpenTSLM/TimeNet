# TimeNet viewer

A read-only browser viewer for a TimeNet registry, or for one pinned dataset version. Launch it
with `timenet view`; [the CLI docs](../../../../../docs/cli/timenet.md) cover usage.

## Layout

- `launch.py` binds `127.0.0.1`, creates the per-launch token, and opens the browser.
- `security.py` is the request boundary: Host and Origin checks, the bearer token, and browser
  security headers.
- `app.py` serves the catalog and manifest API, plus the packaged shell and assets.
- `static/` holds the frontend build. It is committed so the package works without Node. The
  source lives in `frontend/` at the repository root; rebuild it after changing the source:

```bash
cd frontend
npm ci
npm run build
```

## TODO

- **Remote access through an HTTPS reverse proxy.** v0 binds only to `127.0.0.1`, so the first
  review covers a single security model: loopback plus a per-launch token. An earlier revision of
  PR #470 implemented remote access. Bringing it back needs:
  - `--host` for a non-loopback bind, accepted only together with `--public-origin`.
  - `--public-origin`, validated as an HTTPS origin and normalized as browsers do (lowercase
    host, default port dropped, IDNA). It becomes the allowed Host and Origin, and the base of
    the printed URL.
  - Docs requiring the proxy to forward the original `Host` header (for nginx,
    `proxy_set_header Host $host;`), and a threat model for exposing the token beyond loopback.
  - Tests for origin normalization, port mismatches, and invalid binds.
