#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Loading Utility for MagCalc.

This module provides functions to load and validate the spin model
configuration from a YAML file.
"""
import yaml
import logging
import logging
import os
from typing import Dict, Any, List

# Added import for CIF handling
try:
    from . import cif_utils
except ImportError:
    # If using relative import fails (e.g. running script directly), try absolute
    import cif_utils

logger = logging.getLogger(__name__)


def load_spin_model_config(filepath: str) -> Dict[str, Any]:
    """
    Loads and performs basic validation of the spin model configuration
    from a YAML file.

    Args:
        filepath (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration as a Python dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If there's an error parsing the YAML or if essential
                    sections are missing from the configuration.
    """
    logger.info(f"Loading spin model configuration from: {filepath}")
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {filepath}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {filepath}: {e}")
        raise ValueError(f"Invalid YAML format in {filepath}") from e

    # --- CIF File Processing ---
    # Check if a CIF file is specified in the crystal_structure section
    if "crystal_structure" in config and "cif_file" in config["crystal_structure"]:
        cif_rel_path = config["crystal_structure"]["cif_file"]
        # Resolve path relative to the config file location
        base_dir = os.path.dirname(os.path.abspath(filepath))
        cif_abs_path = os.path.join(base_dir, cif_rel_path)
        
        logger.info(f"Detected 'cif_file' in config. Loading structure from: {cif_abs_path}")
        
        # Get magnetic elements filter if provided
        magnetic_elements = config["crystal_structure"].get("magnetic_elements")
        
        try:
            cif_data = cif_utils.load_cif_structure(cif_abs_path, magnetic_elements)
            
            # Override/Merge data
            # Logic: If atoms_uc or unit_cell_vectors are ALREADY in config, 
            # we might want to warn or overwrite? 
            # Standard behavior: Config file 'atoms_uc' takes precedence? 
            # OR CIF takes precedence? 
            # Usually CIF is meant to REPLACE the manual definition.
            
            config["crystal_structure"]["unit_cell_vectors"] = cif_data["unit_cell_vectors"]
            config["crystal_structure"]["atoms_uc"] = cif_data["atoms_uc"]
            
            logger.info("Successfully populated crystal structure from CIF.")
            
        except Exception as e:
            logger.error(f"Failed to load CIF file: {e}")
            raise ValueError(f"Error loading CIF file {cif_abs_path}: {e}") from e
    # --- End CIF File Processing ---

    # Basic validation for essential sections
    required_sections = ["crystal_structure", "interactions", "parameters"]
    for section in required_sections:
        if section not in config:
            msg = f"Missing required section '{section}' in configuration file: {filepath}"
            logger.error(msg)
            raise ValueError(msg)

    logger.info("Spin model configuration loaded and basic validation passed.")
    return config
