"""The wiki generator links only what exists and never touches code or links.

`tools/sync_wiki.py` mirrors TUTORIAL.md and MAGCALC_GUIDE.md to the GitHub wiki.
A mirror that rewrote a code block, or linked a path that is not in the repo, would
publish something the repository does not say, so the rendering is pinned here.
No network: these test the rendering only.
"""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("sync_wiki", REPO / "tools" / "sync_wiki.py")
sync_wiki = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_wiki)

TRACKED = {"GAP_STATUS.md", "magcalc/runner.py", "magcalc/", "examples/",
           "examples/materials/", "examples/materials/FeI2/",
           "examples/materials/FeI2/config_fei2_sun.yaml", "TUTORIAL.md", "MAGCALC_GUIDE.md"}


def test_a_directory_links_to_its_tree():
    out = sync_wiki.render("Configs live in `examples/materials/`.\n", TRACKED)
    assert f"[`examples/materials/`]({sync_wiki.TREE}examples/materials)" in out


def test_an_existing_path_becomes_a_link():
    out = sync_wiki.render("See `GAP_STATUS.md` and `magcalc/runner.py:640`.\n", TRACKED)
    assert f"[`GAP_STATUS.md`]({sync_wiki.BLOB}GAP_STATUS.md)" in out
    assert f"[`magcalc/runner.py:640`]({sync_wiki.BLOB}magcalc/runner.py)" in out


def test_a_bare_package_name_is_not_mistaken_for_its_directory():
    text = "Run `magcalc`, or import `magcalc` from `magcalc/`.\n"
    out = sync_wiki.render(text, TRACKED)
    assert out.count("[`magcalc`]") == 0
    assert f"[`magcalc/`]({sync_wiki.TREE}magcalc)" in out


def test_a_path_that_does_not_exist_is_left_alone():
    text = "Use `calculation.kernels` and `no_such_file.md`.\n"
    assert sync_wiki.render(text, TRACKED) == text


def test_code_blocks_are_never_rewritten():
    text = "```bash\ncat `GAP_STATUS.md`\n```\n"
    assert sync_wiki.render(text, TRACKED) == text


def test_existing_links_are_left_exactly_as_they_are():
    text = "[the record](https://x.org/`GAP_STATUS.md`) and [`GAP_STATUS.md`](GAP_STATUS.md)\n"
    out = sync_wiki.render(text, TRACKED)
    assert out.count("](https://x.org/`GAP_STATUS.md`)") == 1
    assert "[`GAP_STATUS.md`](GAP_STATUS.md)" in out


def test_mirrored_files_link_to_their_wiki_pages():
    out = sync_wiki.render("Read [the guide](MAGCALC_GUIDE.md#9) and `TUTORIAL.md`.\n", TRACKED)
    assert "](AI-Guide#9)" in out
    assert "[`TUTORIAL.md`](Home)" in out


def test_the_real_pages_render_and_carry_the_generated_banner():
    pages = sync_wiki.build_pages(REPO)
    assert set(pages) == {"Home.md", "AI-Guide.md", "_Sidebar.md", "_Footer.md"}
    assert "Edit that file, not this page" in pages["Home.md"]
    # the tutorial's own fenced examples survive byte for byte
    tutorial = (REPO / "TUTORIAL.md").read_text(encoding="utf-8")
    first_fence = tutorial[tutorial.index("```"):tutorial.index("```", tutorial.index("```") + 3) + 3]
    assert first_fence in pages["Home.md"]
