# -*- coding: utf-8 -*-
"""
Utility for generating paths in reciprocal space.
"""
import numpy as np
import logging
from typing import List, Dict, Union
import numpy.typing as npt

logger = logging.getLogger(__name__)


def generate_q_path(
    path_spec: List[str],
    points_per_segment: int,
    high_symmetry_points: Dict[str, Union[List[float], npt.NDArray[np.float_]]],
) -> npt.NDArray[np.float_]:
    """
    Generates a list of q-vectors along a path defined by high-symmetry points.

    Args:
        path_spec (List[str]): A list of names of high-symmetry points defining
            the path segments (e.g., ['Gamma', 'X', 'M', 'Gamma']).
        points_per_segment (int): The number of q-points to generate for each
            segment of the path.
        high_symmetry_points (Dict[str, Union[List[float], npt.NDArray[np.float_]]]):
            A dictionary mapping high-symmetry point names (str) to their
            coordinates (list or array of 3 floats).

    Returns:
        npt.NDArray[np.float_]: A NumPy array of shape (N_total, 3) containing
            the q-vectors along the specified path.

    Raises:
        ValueError: If path_spec is too short, points_per_segment is not positive,
                    or a point name in path_spec is not found in high_symmetry_points.
    """
    if len(path_spec) < 2:
        raise ValueError("path_spec must contain at least two points.")
    if points_per_segment <= 0:
        raise ValueError("points_per_segment must be positive.")

    q_path_segments = []
    num_segments = len(path_spec) - 1

    for i in range(num_segments):
        start_name, end_name = path_spec[i], path_spec[i + 1]
        if (
            start_name not in high_symmetry_points
            or end_name not in high_symmetry_points
        ):
            raise ValueError(
                f"Point name '{start_name}' or '{end_name}' not found in high_symmetry_points."
            )

        start_coord = np.array(high_symmetry_points[start_name], dtype=float)
        end_coord = np.array(high_symmetry_points[end_name], dtype=float)
        # Include endpoint only for the very last segment
        include_endpoint = i == num_segments - 1
        segment_points = np.linspace(
            start_coord, end_coord, points_per_segment, endpoint=include_endpoint
        )
        q_path_segments.append(segment_points)

    return np.vstack(q_path_segments)
