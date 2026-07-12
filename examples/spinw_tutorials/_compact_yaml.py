"""Compact YAML emitter shared by the spinw_tutorials config generators.

Sequences whose elements are all scalars (vectors: positions, offsets,
matrix rows, q-points) are written in flow style `[x, y, z]`; everything
else stays block style, so configs remain diffable but not verbose.
"""
import yaml


class _CompactDumper(yaml.SafeDumper):
    pass


def _represent_list(dumper, data):
    scalars = all(isinstance(x, (int, float, str, bool, type(None))) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data,
                                     flow_style=scalars)


_CompactDumper.add_representer(list, _represent_list)


def dump(data, stream=None, **kwargs):
    kwargs.setdefault("sort_keys", False)
    return yaml.dump(data, stream, Dumper=_CompactDumper, **kwargs)
