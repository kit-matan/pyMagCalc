#!/usr/bin/env python3
"""Regenerate the pyMagCalc GitHub wiki from the files in this repository.

The wiki is a MIRROR, never a second source: every page is generated from a file
that lives in the repo, and the pages say so. Edit the repo file, then run this.

    python tools/sync_wiki.py            # write the pages into the wiki clone
    python tools/sync_wiki.py --check    # exit 1 if the wiki is out of date
    python tools/sync_wiki.py --push     # write, commit and push the wiki

Pages:
    Home.md      <- TUTORIAL.md        (the user reference)
    AI-Guide.md  <- MAGCALC_GUIDE.md   (orientation for AI models and new contributors)
    _Sidebar.md, _Footer.md            (navigation; "generated, do not edit here")

Rendering does two things and nothing else:
  * a back-ticked path that EXISTS in the repo (`GAP_STATUS.md`,
    `examples/materials/FeI2/config_fei2_sun.yaml`) becomes a link to that file on
    GitHub, since a bare path leads nowhere on the wiki;
  * links between the mirrored files (`MAGCALC_GUIDE.md`, `TUTORIAL.md`) point at
    their wiki pages instead.
Fenced code blocks and existing links are left exactly as they are.

The wiki clone defaults to `../pyMagCalc.wiki` beside this repo and is cloned on
first use. GitHub creates a wiki's git repository only when its first page is made
in the web interface, so a brand-new wiki needs that one click before this can push.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OWNER_REPO = "kit-matan/pyMagCalc"
BLOB = f"https://github.com/{OWNER_REPO}/blob/master/"
TREE = f"https://github.com/{OWNER_REPO}/tree/master/"
WIKI_URL = f"https://github.com/{OWNER_REPO}.wiki.git"

# repo file -> wiki page name (without .md)
PAGES = {"TUTORIAL.md": "Home", "MAGCALC_GUIDE.md": "AI-Guide"}

_FENCE = re.compile(r"^(\s*)(```|~~~)")
_LINK = re.compile(r"!?\[[^\]]*\]\([^)]*\)")
_TICK = re.compile(r"`([^`\n]+)`")
_MD_LINK_TARGET = re.compile(r"\]\(([^)#\s]+\.md)(#[^)]*)?\)")


def _tracked_paths(repo: Path) -> set[str]:
    """Tracked files as bare paths, and every directory holding one with a trailing '/'."""
    out = subprocess.run(["git", "-C", str(repo), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    files = set(out.split())
    dirs = {str(p) + "/" for f in files for p in Path(f).parents if str(p) != "."}
    return files | dirs


def _link_path(token: str, tracked: set[str]) -> str | None:
    """The URL for a back-ticked token, if it names a tracked file or directory."""
    path = token.strip()
    if " " in path or path.startswith(("-", "$", "/", ".")) or "://" in path:
        return None
    as_dir = path.endswith("/") or "/" in path.rstrip("/")
    base = re.sub(r":\d+$", "", path).rstrip("/")   # `file.py:123` -> the file
    if base in PAGES:
        return PAGES[base]
    # A bare word such as `magcalc` is a command or package name in prose, not a
    # path: link a file only when it looks like one, a directory only when written
    # as a path.
    if base in tracked and ("/" in base or Path(base).suffix):
        return BLOB + base
    if base + "/" in tracked and as_dir:
        return TREE + base
    return None


def render(text: str, tracked: set[str]) -> str:
    """Rewrite one markdown document for the wiki (see the module docstring)."""
    out, in_fence, fence = [], False, None
    for line in text.splitlines(keepends=True):
        m = _FENCE.match(line)
        if m:
            if not in_fence:
                in_fence, fence = True, m.group(2)
            elif m.group(2) == fence:
                in_fence = False
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        # links between mirrored files -> wiki pages
        line = _MD_LINK_TARGET.sub(
            lambda mm: f"]({PAGES[mm.group(1)]}{mm.group(2) or ''})" if mm.group(1) in PAGES
            else mm.group(0), line)
        # back-ticked repo paths outside existing links -> links
        pieces, pos = [], 0
        for lm in _LINK.finditer(line):
            pieces.append((line[pos:lm.start()], True))
            pieces.append((lm.group(0), False))
            pos = lm.end()
        pieces.append((line[pos:], True))
        rebuilt = []
        for chunk, rewritable in pieces:
            if rewritable:
                def tick(tm):
                    url = _link_path(tm.group(1), tracked)
                    return f"[`{tm.group(1)}`]({url})" if url else tm.group(0)
                chunk = _TICK.sub(tick, chunk)
            rebuilt.append(chunk)
        out.append("".join(rebuilt))
    return "".join(out)


def _source_rev(repo: Path, files) -> str:
    out = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%h", "--", *files],
                         capture_output=True, text=True, check=True).stdout.strip()
    return out or "uncommitted"


def build_pages(repo: Path = REPO) -> dict[str, str]:
    tracked = _tracked_paths(repo)
    rev = _source_rev(repo, list(PAGES))
    pages = {}
    for src, page in PAGES.items():
        body = render((repo / src).read_text(encoding="utf-8"), tracked)
        banner = (f"> *This page is generated from [`{src}`]({BLOB}{src}) in the "
                  f"repository by `tools/sync_wiki.py`. Edit that file, not this page.*\n\n")
        pages[f"{page}.md"] = banner + body
    pages["_Sidebar.md"] = (
        "**pyMagCalc**\n\n"
        "* [Tutorial (Home)](Home)\n"
        "* [AI Guide](AI-Guide)\n\n"
        "**In the repository**\n\n"
        f"* [README]({BLOB}README.md)\n"
        f"* [Validation record (GAP_STATUS)]({BLOB}GAP_STATUS.md)\n"
        f"* [Open work]({BLOB}OPEN_WORK.md)\n"
        f"* [Config rules (CLAUDE.md)]({BLOB}CLAUDE.md)\n\n"
        "**Companions**\n\n"
        "* [fMagCalc (Fortran)](https://github.com/kit-matan/fMagCalc)\n"
        "* [rsMagCalc (Rust)](https://github.com/kit-matan/rsMagCalc)\n")
    pages["_Footer.md"] = (
        f"Generated from pyMagCalc `{rev}` by `tools/sync_wiki.py`. The repository "
        f"files are the source; edits made here are overwritten.\n")
    return pages


def _git(wiki: Path, *args, check=True):
    return subprocess.run(["git", "-C", str(wiki), *args],
                          capture_output=True, text=True, check=check)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wiki-dir", type=Path, default=REPO.parent / "pyMagCalc.wiki")
    ap.add_argument("--check", action="store_true", help="exit 1 if the wiki is stale")
    ap.add_argument("--push", action="store_true", help="commit and push the wiki")
    a = ap.parse_args(argv)

    wiki = a.wiki_dir
    if not (wiki / ".git").exists():
        r = subprocess.run(["git", "clone", "-q", WIKI_URL, str(wiki)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"Cannot clone {WIKI_URL}: {r.stderr.strip()}\n"
                  "A new wiki needs its first page created once in the GitHub web interface "
                  f"(https://github.com/{OWNER_REPO}/wiki).", file=sys.stderr)
            return 2
    else:
        _git(wiki, "pull", "-q", "--ff-only", check=False)
    # Commit as the repo's own identity, never the machine's global one: GitHub credits
    # a commit by its email, and the global identity here belongs to another account.
    for key in ("user.name", "user.email"):
        val = subprocess.run(["git", "-C", str(REPO), "config", "--local", key],
                             capture_output=True, text=True).stdout.strip()
        if val:
            _git(wiki, "config", "--local", key, val)

    pages = build_pages()
    stale = [n for n, body in pages.items()
             if not (wiki / n).exists() or (wiki / n).read_text(encoding="utf-8") != body]
    if a.check:
        print("wiki is up to date" if not stale else "wiki is STALE: " + ", ".join(stale))
        return 1 if stale else 0
    for n in stale:
        (wiki / n).write_text(pages[n], encoding="utf-8")
    print("updated: " + (", ".join(stale) if stale else "nothing (already current)"))
    if a.push and stale:
        _git(wiki, "add", *stale)
        _git(wiki, "commit", "-q", "-m",
             f"Regenerate from pyMagCalc {_source_rev(REPO, list(PAGES))} (tools/sync_wiki.py)")
        r = _git(wiki, "push", "-q", "origin", "HEAD", check=False)
        if r.returncode != 0:
            print(r.stderr.strip(), file=sys.stderr)
            return 3
        print("pushed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
