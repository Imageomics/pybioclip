# SPDX-FileCopyrightText: 2024-present John Bradley <johnbradley2008@gmail.com>
#
# SPDX-License-Identifier: MIT

__all__ = ["TreeOfLifeClassifier", "Rank", "CustomLabelsClassifier", "CustomLabelsBinningClassifier",
           "BIOCLIP_MODEL_STR", "BIOCLIP_V2_MODEL_STR", "BIOCLIP_V1_MODEL_STR"]

# Placeholders so all explicit exports are defined at module scope.
# Real values are loaded lazily in __getattr__.
TreeOfLifeClassifier = None
Rank = None
CustomLabelsClassifier = None
CustomLabelsBinningClassifier = None
BIOCLIP_MODEL_STR = None
BIOCLIP_V2_MODEL_STR = None
BIOCLIP_V1_MODEL_STR = None

# Names resolvable from _constants without loading torch
_CONSTANTS_NAMES = frozenset(["Rank", "BIOCLIP_MODEL_STR", "BIOCLIP_V2_MODEL_STR", "BIOCLIP_V1_MODEL_STR"])


def __getattr__(name):
    if name in _CONSTANTS_NAMES:
        from bioclip._constants import Rank, BIOCLIP_MODEL_STR, BIOCLIP_V2_MODEL_STR, BIOCLIP_V1_MODEL_STR
        g = globals()
        g['Rank'] = Rank
        g['BIOCLIP_MODEL_STR'] = BIOCLIP_MODEL_STR
        g['BIOCLIP_V2_MODEL_STR'] = BIOCLIP_V2_MODEL_STR
        g['BIOCLIP_V1_MODEL_STR'] = BIOCLIP_V1_MODEL_STR
        return g[name]
    if name in __all__:
        from bioclip.predict import TreeOfLifeClassifier, CustomLabelsClassifier, CustomLabelsBinningClassifier
        g = globals()
        g['TreeOfLifeClassifier'] = TreeOfLifeClassifier
        g['CustomLabelsClassifier'] = CustomLabelsClassifier
        g['CustomLabelsBinningClassifier'] = CustomLabelsBinningClassifier
        return g[name]
    raise AttributeError(f"module 'bioclip' has no attribute {name!r}")
