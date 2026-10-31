"""Save and resume for the dimer series' linked-cluster expansion (OPEN_WORK item 10).

`DimerSeriesModel.hopping_series(order)` holds every subtracted cluster weight in
memory until it returns, so a job killed at a 24 h wall used to lose the whole order
(Rb2Cu3SnF12 at order 9: 910 371 clusters, about a day on 120 threads). With a
checkpoint directory the memo is written to disk as it grows, one file per chunk of
new topologies, and a second job with the same directory loads it and continues.

Layout, one subdirectory per order:

    <dir>/order-<n>/manifest.json   what the checkpoint belongs to (JSON, readable)
    <dir>/order-<n>/w-000000.pkl    the memo entries of one chunk, append only
    <dir>/order-<n>/lock            held (flock) by the process writing it

What is saved is the memo itself -- each topology's subtracted weight
(gs_w, Lc0, delta_trim), its structural key, one embedding and its additivity
ratio -- so a resumed run sums exactly the numbers an uninterrupted one would, and
its hoppings are bit for bit the same (tests/test_dimer_series_checkpoint.py).

Every file is written to a temporary name, fsynced and renamed, so a kill mid-write
leaves the previous state intact. The manifest and every chunk file carry a
fingerprint of everything that decides a memoized weight: the local levels and their
Sz labels (whether the PT runs in Sz sectors), every link's pair operator (DM
included) and cell offset, the lattice and unit positions, the order, the kernel set,
the cluster enumeration, and the version of this format and of the algorithm. A
mismatch refuses to resume and names what differs; nothing is reused silently.

The chunk files are pickles that hold only tuples, bytes, ints, floats and strings
(arrays as raw bytes), read back by an unpickler that refuses every global, so loading
one cannot run code.
"""
import hashlib
import json
import logging
import os
import pickle
import re
import time

import numpy as np

logger = logging.getLogger(__name__)

FORMAT = "magcalc-dimer-series-checkpoint/1"
# Bump whenever a change to the per-cluster PT, the subtraction or the structural key
# changes a memoized weight: a checkpoint written before it must then be refused.
ALGORITHM = "2026-10-31: projective cluster-additive T0 sector (pyMagCalc bfae055)"
# sysexits EX_TEMPFAIL ("temporary failure; try again"): a job that ends with it saved
# its progress and wants the next one in the chain
EXIT_INTERRUPTED = 75
ENV = "MAGCALC_SERIES_CHECKPOINT"

_CHUNK = re.compile(r"^w-(\d{6,})\.pkl$")


class CheckpointMismatch(ValueError):
    """The checkpoint directory belongs to a different model, order or code."""


class SeriesInterrupted(SystemExit):
    """A stop signal arrived; the finished chunks are on disk. A SystemExit, so an
    unhandled one ends the process with status EXIT_INTERRUPTED (75), and no
    `except Exception` on the way up swallows it."""

    def __init__(self, message, order=None, directory=None, signum=None):
        super().__init__(EXIT_INTERRUPTED)
        self.message = message
        self.order = order
        self.directory = directory
        self.signum = signum

    def __str__(self):
        return self.message


def _hex(*chunks):
    h = hashlib.blake2b(digest_size=16)
    for c in chunks:
        h.update(c if isinstance(c, bytes) else str(c).encode())
        h.update(b"\x00")
    return h.hexdigest()


def _arr(x, dtype):
    a = np.ascontiguousarray(x, dtype=dtype)
    return f"{a.dtype.str}{a.shape}".encode() + a.tobytes()


def model_parts(model, order):
    """{part: digest} of everything that decides a memoized weight at `order`
    (the enumeration is checked separately: it needs the enumerated levels)."""
    links = [_hex(int(u), int(v), tuple(int(x) for x in off), _arr(V, complex))
             for (u, v, off, V) in model.link_types]
    sz = model.sz_levels
    return {
        "format": FORMAT,
        "algorithm": ALGORITHM,
        "order": int(order),
        "kernels": str(model.kernels),
        "units": len(model.levels),
        "nloc": int(model.nloc),
        "link_types": len(model.link_types),
        "sz_sectors": sz is not None,
        "levels": _hex(*(_arr(l, float) for l in model.levels)),
        "sz_levels": _hex(*(_arr(s, float) for s in sz)) if sz is not None else None,
        "links": _hex(*links),
        "lattice": _hex(_arr(model.lat, float), _arr(model.pos, float)),
    }


def enumeration_parts(levels):
    """Per cluster size: (count, order-independent digest of the canonical forms)."""
    out = {}
    for size, forms in sorted(levels.items()):
        acc = 0
        for f in forms:
            acc = (acc + int.from_bytes(hashlib.blake2b(repr(f).encode(),
                                                        digest_size=16).digest(),
                                        "little")) % (1 << 128)
        out[str(size)] = [len(forms), f"{acc:032x}"]
    return out


def _fingerprint(parts, enum):
    return _hex(json.dumps(parts, sort_keys=True), json.dumps(enum, sort_keys=True))


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"a dimer-series checkpoint holds no objects, got {module}.{name}")


def _spec(a):
    a = np.ascontiguousarray(a)
    return (a.dtype.str, a.shape, a.tobytes())


def _unspec(s):
    dt, shape, raw = s
    return np.frombuffer(raw, dtype=np.dtype(dt)).reshape(shape)


def _write_atomic(path, data):
    d, name = os.path.split(path)
    tmp = os.path.join(d, f".{name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    try:                       # make the rename itself durable
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


class SeriesCheckpoint:
    """One order's checkpoint directory, opened for loading and appending."""

    def __init__(self, root, model, order, levels):
        self.root = os.path.abspath(os.fspath(root))
        self.dir = os.path.join(self.root, f"order-{int(order)}")
        os.makedirs(self.dir, exist_ok=True)
        self._lock_fh = None
        self._lock()
        parts = model_parts(model, order)
        enum = enumeration_parts(levels)
        self.fingerprint = _fingerprint(parts, enum)
        man = os.path.join(self.dir, "manifest.json")
        if os.path.exists(man):
            with open(man) as fh:
                old = json.load(fh)
            if old.get("fingerprint") != self.fingerprint:
                self.close()
                raise CheckpointMismatch(self._mismatch(old, parts, enum))
        else:
            info = dict(fingerprint=self.fingerprint, model=parts, enumeration=enum,
                        created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                        magcalc=_provenance())
            _write_atomic(man, (json.dumps(info, indent=1) + "\n").encode())
        for name in os.listdir(self.dir):           # a temp file is a write that died
            if name.startswith(".w-") and name.endswith(".tmp"):
                os.remove(os.path.join(self.dir, name))
        self._next = 1 + max((int(m.group(1)) for m in map(_CHUNK.match,
                                                             os.listdir(self.dir)) if m),
                             default=-1)

    def _mismatch(self, old, parts, enum):
        diff = [k for k in sorted(set(parts) | set(old.get("model", {})))
                if old.get("model", {}).get(k) != parts.get(k)]
        if old.get("enumeration") != enum:
            diff.append("enumeration")
        detail = ", ".join(f"{k}: saved {old.get('model', {}).get(k)!r}, now "
                           f"{parts.get(k)!r}" for k in diff if k in parts)
        return (f"dimer-series checkpoint {self.dir} belongs to a different "
                f"calculation (differs in: {', '.join(diff) or 'fingerprint'}"
                f"{'; ' + detail if detail else ''}). Refusing to reuse it: give "
                f"another directory, or delete this one to start over.")

    def _lock(self):
        try:
            import fcntl
        except ImportError:                       # not POSIX
            return
        fh = open(os.path.join(self.dir, "lock"), "a")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            raise RuntimeError(f"dimer-series checkpoint {self.dir} is in use by "
                               f"another process")
        except OSError as e:                      # e.g. Lustre mounted without flock
            logger.warning("dimer-series checkpoint %s: cannot lock (%s); make sure "
                           "no other job writes to it", self.dir, e)
        self._lock_fh = fh

    def close(self):
        if self._lock_fh is not None:
            self._lock_fh.close()                 # releases the flock
            self._lock_fh = None

    def files(self):
        return sorted((n for n in os.listdir(self.dir) if _CHUNK.match(n)),
                      key=lambda n: int(_CHUNK.match(n).group(1)))

    def load(self):
        """Every saved entry: [(skey, form, gs_w, Lc0, delta_trim, ratio)], arrays
        read only (they are views of the file's bytes)."""
        out = []
        for name in self.files():
            path = os.path.join(self.dir, name)
            try:
                with open(path, "rb") as fh:
                    fmt, fp, entries = _NoGlobals(fh).load()
            except Exception as e:
                raise CheckpointMismatch(f"dimer-series checkpoint file {path} cannot "
                                         f"be read ({e})") from e
            if fmt != FORMAT or fp != self.fingerprint:
                raise CheckpointMismatch(f"dimer-series checkpoint file {path} belongs "
                                         f"to a different calculation than "
                                         f"{self.dir}/manifest.json")
            for skey, form, gs, Lc0, delta, ratio in entries:
                out.append((skey, form, _unspec(gs), Lc0, _unspec(delta), ratio))
        return out

    def write(self, entries):
        """Append one file of [(skey, form, gs_w, Lc0, delta_trim, ratio)]."""
        if not entries:
            return
        data = pickle.dumps((FORMAT, self.fingerprint,
                             [(skey, form, _spec(gs), int(Lc0), _spec(delta), float(r))
                              for skey, form, gs, Lc0, delta, r in entries]),
                            protocol=5)
        _write_atomic(os.path.join(self.dir, f"w-{self._next:06d}.pkl"), data)
        self._next += 1


def _provenance():
    try:
        from .. import provenance
        return provenance.describe()
    except Exception:
        return None
