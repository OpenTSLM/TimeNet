"""Post-build enhancer that makes the built site agent-friendly and social-ready.

Runs after ``zensical build`` and works on the ``site/`` output plus the ``docs/`` source:

1. Mirrors every source Markdown page into ``site/`` (front matter stripped), so appending
   ``.md`` to a page URL returns clean Markdown (``…/client.html`` -> ``…/client.md``).
2. Writes ``site/llms.txt`` (an llmstxt.org discovery index) and ``site/llms-full.txt`` (every
   page concatenated), both linking the raw ``.md`` URLs.
3. Injects Open Graph / Twitter meta tags and a schema.org JSON-LD ``Article`` block (authored by
   TimeNet / OpenTSLM) into each HTML page.
4. Adds a "Copy as Markdown" button that fetches the page's sibling ``.md`` mirror.

Run it with the docs env::

    uv run --group docs --extra build python scripts/gen_site_extras.py
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tomllib
from urllib.parse import quote

from bs4 import BeautifulSoup
import griffe
from markdownify import markdownify
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
SITE_DIR = REPO_ROOT / "site"
CONFIG_FILE = REPO_ROOT / "zensical.toml"
SCHEMAS_DIR = REPO_ROOT / "packages" / "timenet" / "src" / "timenet" / "schemas"

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n+", re.DOTALL)

_CHEVRON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="m6 9 6 6 6-6"/></svg>'
)


def _copy_control_html(site_url: str, rel: str) -> str:
    """Build the "Copy page" split-button dropdown markup for one page.

    Args:
        site_url: The site root URL (no trailing slash).
        rel: The page path relative to ``site/`` (e.g. ``api/client.html``).

    Returns:
        The control's HTML. Behavior (clipboard, menu, keyboard) is wired by ``extra.js``;
        the copy action reads the sibling ``.md`` relative to the current URL, while the
        "Open in ..." links use the page's absolute ``.md`` URL so external LLMs can fetch it.
    """
    base = rel[:-5] if rel.endswith(".html") else rel
    md_name = base.rsplit("/", 1)[-1] + ".md"
    md_abs = f"{site_url}/{base}.md"
    q = quote(f"Read {md_abs} so I can ask you questions about it.")
    sid = base.replace("/", "-")
    return (
        '<div class="copy-page">'
        '<button class="copy-page__main" type="button" data-copy>Copy page</button>'
        f'<button class="copy-page__toggle" type="button" id="cp-toggle-{sid}" aria-haspopup="menu" '
        f'aria-expanded="false" aria-controls="cp-menu-{sid}" aria-label="More copy options">{_CHEVRON}</button>'
        f'<ul class="copy-page__menu" id="cp-menu-{sid}" role="menu" aria-labelledby="cp-toggle-{sid}" hidden>'
        '<li role="none"><button role="menuitem" type="button" data-copy>'
        '<span class="copy-page__title">Copy as Markdown</span></button></li>'
        f'<li role="none"><a role="menuitem" href="./{md_name}">'
        '<span class="copy-page__title">View as Markdown</span></a></li>'
        f'<li role="none"><a role="menuitem" href="https://chatgpt.com/?q={q}" target="_blank" rel="noopener">'
        '<span class="copy-page__title">Open in ChatGPT</span></a></li>'
        f'<li role="none"><a role="menuitem" href="https://claude.ai/new?q={q}" target="_blank" rel="noopener">'
        '<span class="copy-page__title">Open in Claude</span></a></li>'
        "</ul>"
        '<span class="sr-only" role="status" aria-live="polite"></span>'
        "</div>"
    )


def _load_config() -> dict:
    """Parse ``zensical.toml`` and return its ``[project]`` scope.

    The native Zensical config nests everything under ``[project]``; nav parses to the same
    list-of-mappings the YAML config used, so the rest of this script is unchanged.

    Returns:
        The project configuration mapping (``site_url``, ``site_name``, ``nav``, ...).
    """
    data = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return data.get("project", data)


def _split_front_matter(text: str) -> tuple[dict, str]:
    """Split a Markdown document into its front-matter mapping and body.

    Args:
        text: The raw Markdown text.

    Returns:
        A ``(metadata, body)`` pair; ``metadata`` is empty when there is no front matter.
    """
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta = yaml.safe_load(match.group(1)) or {}
    return meta, text[match.end() :]


def _title_of(body: str) -> str:
    """Return the first ``# `` heading of a Markdown body, or an empty string."""
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _rendered_markdown(html_path: Path) -> str | None:
    """Convert a built page's main content back to Markdown, or ``None`` if unavailable.

    Used for mkdocstrings pages whose source is only a ``:::`` directive, so the ``.md`` mirror
    carries the rendered API reference instead of the unexpanded directive.

    Args:
        html_path: The built ``site/`` HTML file for the page.

    Returns:
        The content as Markdown, or ``None`` when the file or content region is missing.
    """
    if not html_path.exists():
        return None
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    inner = soup.select_one(".md-content__inner")
    if inner is None:
        return None
    for junk in inner.select("a.headerlink, .copy-page"):
        junk.decompose()
    return markdownify(str(inner), heading_style="ATX").strip() + "\n"


def _mirror_markdown() -> None:
    """Write a clean ``.md`` for every page into ``site/`` (front matter stripped).

    For pages whose source body is only a mkdocstrings ``:::`` directive, the built HTML is
    converted back to Markdown so the mirror holds the rendered reference, not the directive.
    """
    for md in DOCS_DIR.rglob("*.md"):
        rel = md.relative_to(DOCS_DIR)
        _meta, body = _split_front_matter(md.read_text(encoding="utf-8"))
        if body.lstrip().startswith(":::") and (rendered := _rendered_markdown(SITE_DIR / rel.with_suffix(".html"))):
            body = rendered
        dest = SITE_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")


def _nav_pages(nav: list) -> list[tuple[str | None, str, str]]:
    """Flatten a mkdocs ``nav`` into ``(section, title, docs_relpath)`` triples.

    Args:
        nav: The ``nav`` list from ``zensical.toml``.

    Returns:
        One entry per page, in nav order; ``section`` is the top-level group name or ``None``.
    """
    pages: list[tuple[str | None, str, str]] = []

    def walk(items: list, section: str | None) -> None:
        for item in items:
            if isinstance(item, str):
                pages.append((section, "", item))
            elif isinstance(item, dict):
                for title, value in item.items():
                    if isinstance(value, str):
                        pages.append((section, title, value))
                    elif isinstance(value, list):
                        walk(value, title)

    walk(nav, None)
    return pages


def _page_description(relpath: str) -> str:
    """Return the front-matter description of a docs page, or an empty string.

    Args:
        relpath: The page path relative to ``docs/``.

    Returns:
        The ``description`` field, or ``""`` when the page or field is missing.
    """
    md = DOCS_DIR / relpath
    if not md.exists():
        return ""
    meta, _body = _split_front_matter(md.read_text(encoding="utf-8"))
    return str(meta.get("description", ""))


def _md_url(site_url: str, relpath: str) -> str:
    """Return the absolute URL of a page's raw Markdown mirror."""
    return f"{site_url}/{relpath}"


def _generate_llms(config: dict) -> None:
    """Write ``site/llms.txt`` and ``site/llms-full.txt`` from the nav and page front matter.

    Args:
        config: The parsed ``zensical.toml`` project configuration.
    """
    site_url = str(config.get("site_url", "")).rstrip("/")
    name = config.get("site_name", "Documentation")
    description = config.get("site_description", "")
    pages = _nav_pages(config.get("nav") or [])

    lines = [f"# {name}", "", f"> {description}", ""]
    current: str | None = "__start__"
    for section, title, relpath in pages:
        if section != current:
            current = section
            if section:
                lines += ["", f"## {section}", ""]
        label = title or _title_of(DOCS_DIR.joinpath(relpath).read_text(encoding="utf-8")) or relpath
        desc = _page_description(relpath)
        suffix = f": {desc}" if desc else ""
        lines.append(f"- [{label}]({_md_url(site_url, relpath)}){suffix}")
    (SITE_DIR / "llms.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    full = [f"# {name}", "", f"> {description}", ""]
    for _section, _title, relpath in pages:
        mirror = SITE_DIR / relpath  # the mirrored .md (rendered for mkdocstrings pages)
        body = mirror.read_text(encoding="utf-8").strip() if mirror.exists() else ""
        full += ["", "---", "", body, ""]
    (SITE_DIR / "llms-full.txt").write_text("\n".join(full) + "\n", encoding="utf-8")


def _meta_tag(soup: BeautifulSoup, attr: str, key: str, content: str) -> None:
    """Append a ``<meta>`` tag to ``<head>`` with the given attribute and content."""
    if soup.head is not None:
        soup.head.append(soup.new_tag("meta", attrs={attr: key, "content": content}))


_API_ROOTS = ("timenet", "timenet_connectors")


def _load_griffe() -> dict:
    """Load the API packages with Griffe for source-location resolution.

    Returns:
        A mapping of top-level package name to its loaded Griffe module.
    """
    search = [
        str(REPO_ROOT / "packages" / "timenet" / "src"),
        str(REPO_ROOT / "packages" / "timenet-connectors" / "src"),
    ]
    return {root: griffe.load(root, search_paths=search) for root in _API_ROOTS}


def _source_url(index: dict, repo_url: str, qualified: str) -> str | None:
    """Return the GitHub blob URL for a qualified symbol id, or ``None`` if it can't be resolved.

    Args:
        index: The Griffe package index from :func:`_load_griffe`.
        repo_url: The repository URL (e.g. ``https://github.com/OpenTSLM/TimeNet``).
        qualified: The dotted symbol path (e.g. ``timenet.client.TimeNet.load``).

    Returns:
        A ``.../blob/main/<path>#L<line>`` URL, or ``None``.
    """
    top, _, rest = qualified.partition(".")
    module = index.get(top)
    if module is None:
        return None
    try:
        obj = module[rest] if rest else module
        filepath = obj.filepath
        line = obj.lineno
    except Exception:  # unresolved / aliased symbols are simply skipped
        return None
    if filepath is None:
        return None
    resolved = Path(filepath).resolve()
    if REPO_ROOT not in resolved.parents:
        return None  # symbol inherited from a dependency: its source lives outside the repo, so no link
    rel = resolved.relative_to(REPO_ROOT)
    anchor = f"#L{line}" if line else ""
    return f"{repo_url}/blob/main/{rel.as_posix()}{anchor}"


def _add_source_links(soup: BeautifulSoup, index: dict, repo_url: str) -> None:
    """Append a GitHub "source" link to each mkdocstrings symbol heading on an API page.

    Args:
        soup: The parsed page.
        index: The Griffe package index.
        repo_url: The repository URL.
    """
    for heading in soup.select(".doc-heading[id]"):
        qualified = str(heading.get("id") or "")
        if qualified.partition(".")[0] not in _API_ROOTS or heading.select_one(".doc-source-link"):
            continue
        url = _source_url(index, repo_url, qualified)
        if not url:
            continue
        link = soup.new_tag(
            "a",
            href=url,
            attrs={"class": "doc-source-link", "target": "_blank", "rel": "noopener", "title": "View source on GitHub"},
        )
        link.string = "source"
        heading.append(link)


def _enhance_html(config: dict) -> None:  # noqa: PLR0914
    """Inject social meta tags, JSON-LD, a copy button, and API source links into every page.

    Args:
        config: The parsed ``zensical.toml`` project configuration.
    """
    site_url = str(config.get("site_url", "")).rstrip("/")
    repo_url = str(config.get("repo_url", "")).rstrip("/")
    name = config.get("site_name", "Documentation")
    image = f"{site_url}/assets/social-card.png"
    griffe_index = _load_griffe() if repo_url else {}
    # The JSON-LD author org is the repo owner, derived from repo_url so a rename only touches config.
    author_name = repo_url.rsplit("/", 2)[-2] if repo_url else name
    author_url = repo_url.rsplit("/", 1)[0] if repo_url else site_url

    for html in SITE_DIR.rglob("*.html"):
        rel = html.relative_to(SITE_DIR).as_posix()
        if rel == "404.html":
            continue
        soup = BeautifulSoup(html.read_text(encoding="utf-8"), "html.parser")
        if not soup.head or soup.find("meta", attrs={"property": "og:title"}):
            continue

        page_url = f"{site_url}/{rel}"
        title = name
        if soup.h1:
            texts = [
                str(node)
                for node in soup.h1.find_all(string=True)
                if node.parent is None or "headerlink" not in (node.parent.get("class") or [])
            ]
            title = "".join(texts).strip() or title
        elif soup.title and soup.title.string:
            title = soup.title.string
        description = _page_description(rel.replace(".html", ".md"))
        if not description and (p := soup.select_one(".md-content__inner p")):
            description = p.get_text(strip=True)

        if not soup.find("meta", attrs={"name": "description"}):
            _meta_tag(soup, "name", "description", description)
        for prop, content in (
            ("og:type", "website"),
            ("og:title", title),
            ("og:description", description),
            ("og:url", page_url),
            ("og:image", image),
            ("twitter:card", "summary_large_image"),
            ("twitter:title", title),
            ("twitter:description", description),
            ("twitter:image", image),
        ):
            _meta_tag(soup, "property", prop, content)

        ld = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": description,
            "url": page_url,
            "image": image,
            "author": {"@type": "Organization", "name": author_name, "url": author_url},
            "publisher": {"@type": "Organization", "name": name},
        }
        script = soup.new_tag("script", type="application/ld+json")
        # Escape "</" so a title/description containing "</script>" can't break out of the tag.
        script.string = json.dumps(ld).replace("</", "<\\/")
        soup.head.append(script)  # head is non-None past the guard above

        if (inner := soup.select_one(".md-content__inner")) and not soup.select_one(".copy-page"):
            control = BeautifulSoup(_copy_control_html(site_url, rel), "html.parser")
            inner.insert(0, control)

        if griffe_index and rel.startswith("api/"):
            _add_source_links(soup, griffe_index, repo_url)

        html.write_text(str(soup), encoding="utf-8")


def _copy_schemas() -> None:
    """Publish the packaged JSON Schemas under ``site/schemas/`` at their ``$id`` basename.

    The ``timenet`` package ships the schemas but loads them locally (``importlib.resources``), so the
    ``$id`` URLs (``.../schemas/<name>-v1.schema.json``) resolve to nothing on their own. Copying each
    schema to the filename its ``$id`` ends in makes those URLs serve for real, so external validators
    and IDEs can fetch them.
    """
    dest_dir = SITE_DIR / "schemas"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(SCHEMAS_DIR.glob("*.schema.json")):
        schema = json.loads(src.read_text(encoding="utf-8"))
        name = str(schema["$id"]).rsplit("/", 1)[-1]  # e.g. manifest-v1.schema.json
        shutil.copyfile(src, dest_dir / name)


def main() -> None:
    """Run all post-build enhancements against ``site/``.

    Raises:
        SystemExit: If ``site/`` does not exist (the build has not run yet).
    """
    if not SITE_DIR.exists():
        raise SystemExit("site/ not found; run `zensical build` first.")
    config = _load_config()
    _mirror_markdown()
    _generate_llms(config)
    _enhance_html(config)
    _copy_schemas()
    print("Enhanced site: .md mirrors, llms.txt, llms-full.txt, social meta, JSON-LD, copy button, schemas")


if __name__ == "__main__":
    main()
