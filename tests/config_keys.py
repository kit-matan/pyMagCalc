"""Enumerate the config keys the CODE reads, by parsing it (OPEN_WORK item 5).

THE POINT, AND WHY THE OBVIOUS VERSION DOES NOT WORK. The 2026-08-04 coverage audit
swept the DOCUMENTED keys against `tests/` and found 11 with no test. It could not
find `calculation.imaginary_rel_tolerance`, which was in neither the docs nor the
tests -- invisible to the very process meant to find gaps, because the process
started from the documentation. A key that nobody wrote down and nobody tested is
exactly the one that rots.

So this starts from the source instead. It walks the package with `ast` and records
every `<block>.get("key")` / `<block>["key"]` where `<block>` is a local bound to a
config section (`cfg = final_config.get("scga", {})` and friends). That is a
heuristic, not a parser for a language the config does not have -- but it is a
heuristic that fails in the SAFE direction: a key it cannot see is one nothing else
was going to see either, while a spurious entry shows up as a loud "documented
nowhere" and gets deleted from the list by hand.

Used by `tests/test_config_key_coverage.py`; importable on its own to print the
table:

    python -c "import sys; sys.path.insert(0,'tests'); import config_keys; \
               config_keys.report()"
"""
import ast
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PKG = os.path.join(ROOT, "magcalc")

# Names that hold the WHOLE config in the sources scanned. A block is discovered as
# `<something> = <one of these>.get("block", ...)`.
CONFIG_NAMES = {"config", "final_config", "cfg", "full_config", "self.config"}

# Top-level blocks whose keys are user-facing. Anything read off the whole config
# with one of these names IS a block; other `.get` calls on it are top-level keys.
BLOCKS = {
    "calculation", "tasks", "plotting", "minimization", "output", "q_path",
    "scga", "thermal_mc", "wang_landau", "sampled_correlations",
    "sun_sampled_correlations", "kpm", "corrections", "energy_cut", "fitting",
    "powder_average", "magnetic_structure", "crystal_structure", "interactions",
    "mcif", "parameters",
}


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _read_key(node):
    """('name', 'key') if `node` is `name.get("key" ...)` or `name["key"]`."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "get" and node.args:
        key = _const_str(node.args[0])
        if key and isinstance(node.func.value, ast.Name):
            return node.func.value.id, key
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        key = _const_str(node.slice)
        if key:
            return node.value.id, key
    return None


def keys_from_source(path):
    """{block: {key, ...}} read by one module."""
    with open(path) as handle:
        try:
            tree = ast.parse(handle.read(), filename=path)
        except SyntaxError:
            return {}

    # Pass 1: which locals hold which block?
    holders = {}                       # local name -> block name
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        # strip a trailing `or {}`
        if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
            value = value.values[0]
        got = _read_key(value)
        if got and got[0] in CONFIG_NAMES and got[1] in BLOCKS:
            holders[target.id] = got[1]

    # Pass 2: keys read off those locals, plus keys read straight off the config.
    out = {}
    for node in ast.walk(tree):
        got = _read_key(node)
        if not got:
            continue
        name, key = got
        if name in holders:
            out.setdefault(holders[name], set()).add(key)
        elif name in CONFIG_NAMES and key not in BLOCKS:
            out.setdefault("<top-level>", set()).add(key)
    return out


def code_keys():
    """{block: {key, ...}} over the whole package."""
    merged = {}
    for path in sorted(glob.glob(os.path.join(PKG, "**", "*.py"), recursive=True)):
        for block, keys in keys_from_source(path).items():
            merged.setdefault(block, set()).update(keys)
    return merged


# The audit's OWN files are excluded from the sweep. Without this the allow-list is
# self-fulfilling: `test_config_key_coverage.py` names every key it excuses, the text
# sweep then finds those names, and each excused key reads as covered -- so the audit
# would certify exactly the keys it was told to ignore.
SELF = {"test_config_key_coverage.py", "config_keys.py"}


def _text_of(paths):
    blob = []
    for path in paths:
        try:
            with open(path, errors="ignore") as handle:
                blob.append(handle.read())
        except OSError:
            pass
    return "\n".join(blob)


def exercised_keys():
    """Every identifier-looking token appearing in the shipped configs or in tests.

    Deliberately a TEXT sweep and not a YAML walk: a key is "exercised" if a config
    sets it or a test names it, and a test names it in Python (`scga={"mesh_density":
    ...}`), not only in YAML.
    """
    blob = _text_of(
        sorted(glob.glob(os.path.join(ROOT, "examples", "**", "*.yaml"), recursive=True))
        + sorted(glob.glob(os.path.join(ROOT, "examples", "**", "*.py"), recursive=True))
        + [q for q in sorted(glob.glob(os.path.join(HERE, "test_*.py")))
           if os.path.basename(q) not in SELF]
    )
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", blob))


def uncovered():
    """{block: sorted[key]} for keys the code reads that nothing exercises."""
    exercised = exercised_keys()
    out = {}
    for block, keys in sorted(code_keys().items()):
        missing = sorted(k for k in keys if k not in exercised)
        if missing:
            out[block] = missing
    return out


def report():
    found = code_keys()
    print(f"{sum(len(v) for v in found.values())} config keys read by the code, "
          f"in {len(found)} blocks")
    miss = uncovered()
    if not miss:
        print("every one of them appears in a shipped config or a test.")
        return
    print("\nNOT exercised by any config or test:")
    for block, keys in miss.items():
        print(f"  {block}: {', '.join(keys)}")


if __name__ == "__main__":
    report()
