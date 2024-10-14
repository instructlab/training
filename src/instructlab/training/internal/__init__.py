# SPDX-License-Identifier: Apache-2.0

"""
This module is intended to house INTERNAL functions & symbols. Things in this module are considered
as private API and are thus not supported.

The signatures here are also not considered in the versioning scheme under semver - and 
may change at any time.

By using this module, you assume full responsibility of maintenance.
"""

__all__ = ("__SuperAccelerator")

from .accelerator import __SuperAccelerator