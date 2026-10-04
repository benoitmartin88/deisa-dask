###################################################################################################
# Copyright (c) 2026 Commissariat a l'énergie atomique et aux énergies alternatives (CEA)
# SPDX-License-Identifier: MIT
###################################################################################################
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deisa-dask")  # installed version
except PackageNotFoundError:
    from .__version__ import __version__  # fallback

from .bridge import Bridge
from .deisa import Deisa
from .mergeable_pca import (
    VELOCITY_AXES,
    MergeablePCA,
    PCASummary,
    local_pca,
    local_pca_from_chunk,
    merge_pca,
    merge_tree,
)
from .utils import get_connection_info

__all__ = [
    "VELOCITY_AXES",
    "Bridge",
    "Deisa",
    "MergeablePCA",
    "PCASummary",
    "__version__",
    "get_connection_info",
    "local_pca",
    "local_pca_from_chunk",
    "merge_pca",
    "merge_tree",
]
