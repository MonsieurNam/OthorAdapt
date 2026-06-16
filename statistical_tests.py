"""Documented statistical report format for revision run manifests.

The aggregation pipeline reports, per dataset/shot/method/split:
- n
- mean_accuracy
- std_accuracy (sample standard deviation)
- ci95_half_width (two-sided t interval)

When a baseline method is supplied, paired rows additionally report:
- paired_n
- paired_seeds
- mean_delta
- std_delta
- ci95_delta_half_width
- effect_size_dz (paired Cohen dz)
"""

from aggregate_results import aggregate_manifest


__all__ = ["aggregate_manifest"]
