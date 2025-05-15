"""
Some Attributions for the DINA project.

This module contains some of the attribution methods used in the DINA project.
They have been adapted from the original code (but optimized for this repo) from
the Xplique library:
https://github.com/deel-ai/xplique/tree/master
"""
from .lime import Lime
from .global_sensitivity_analysis import HsicAttributionMethod, SobolAttributionMethod