"""
Task-Discriminant Component Analysis (TDCA) for 12JFPM SSVEP.

TDCA extends TRCA with lagging augmentation and DSP (Discriminant
Spatial Pattern) projection. This is a complex training-based method.

STATUS: DEFERRED.

TDCA requires substantial adaptation effort:
  - DSP kernel (between-class / within-class scatter with nearestPD)
  - Lagging augmentation with sin/cos projection matrices
  - Filter bank integration
  - Multi-class spatial filter decomposition

is 476 lines with tight coupling to the upstream opt/config system and
a data layout that differs significantly from our [N, T, C] contract.

Porting this correctly requires dedicated effort beyond this phase.
A clean TODO is left for future implementation.

See project_context/NEXT_TASK.md for the follow-up item.
"""

# Key components to adapt:
#   - xiang_dsp_kernel (between/within-class scatter)
#   - xiang_dsp_feature (spatial feature extraction)
#   - lagging_aug (delayed augmentation + projection)
#   - proj_ref (sin/cos QR projection matrices)
#   - filter_bank (10-band Chebyshev, same as TRCA/FBCCA)
#   - fit/predict interface matching TRCAModel style
