"""The GUI backend `/parse-mcif` route (Studio web + native mCIF import).

Pins the route to the same TbSb reference as tests/test_mcif.py: it must return
the fully-expanded magnetic cell as EXPLICIT atoms (P1) plus a `generic` magnetic
structure carrying the per-site spin directions -- the shape both Studio clients
consume. Skips cleanly if the web stack (fastapi/httpx) is not installed.
"""
import os

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

HERE = os.path.dirname(__file__)
MCIF = os.path.join(HERE, "data_TbSb.mcif")


@pytest.fixture(scope="module")
def client():
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "gui"))
    import server  # gui/server.py
    return TestClient(server.app)


def test_parse_mcif_returns_expanded_explicit_magnetic_cell(client):
    with open(MCIF, "rb") as f:
        r = client.post("/parse-mcif", files={"file": ("data_TbSb.mcif", f)})
    assert r.status_code == 200, r.text
    j = r.json()

    # Fully expanded magnetic cell -> explicit atoms in P1 (no further symmetry).
    assert j["atom_mode"] == "explicit"
    assert j["lattice"]["space_group"] == 1
    assert j["n_sites"] == len(j["wyckoff_atoms"]) == 6      # TbSb G-type magnetic cell
    assert j["magnetic_elements"] == ["Tb"]                  # element symbol, not "Tb1_1"

    ms = j["magnetic_structure"]
    assert ms["enabled"] is True and ms["pattern_type"] == "generic"
    dirs = np.array(ms["directions"])
    assert dirs.shape == (6, 3)
    # G-type antiferromagnet: moments along +/- c, net zero.
    assert np.allclose(dirs.sum(axis=0), 0, atol=1e-6)
    assert np.allclose(np.abs(dirs), np.array([[0, 0, 1]] * 6), atol=1e-6)


def test_parse_mcif_spin_s_form_field(client):
    with open(MCIF, "rb") as f:
        r = client.post("/parse-mcif", files={"file": ("t.mcif", f)},
                        data={"spin_s": "2.5"})
    assert r.status_code == 200, r.text
    assert all(a["spin_S"] == 2.5 for a in r.json()["wyckoff_atoms"])
