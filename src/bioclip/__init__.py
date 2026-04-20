# SPDX-FileCopyrightText: 2024-present John Bradley <johnbradley2008@gmail.com>
#
# SPDX-License-Identifier: MIT

from bioclip._constants import (
    Rank,
    BIOCLIP_MODEL_STR, BIOCLIP_V2_MODEL_STR, BIOCLIP_V1_MODEL_STR,
)

__all__ = [
    "TreeOfLifeClassifier", "CustomLabelsClassifier", "CustomLabelsBinningClassifier",
    Rank,
    BIOCLIP_MODEL_STR, BIOCLIP_V2_MODEL_STR, BIOCLIP_V1_MODEL_STR,
]

def __getattr__(name):
    if name in __all__:
        from bioclip.predict import TreeOfLifeClassifier, CustomLabelsClassifier, CustomLabelsBinningClassifier
        g = globals()
        g['TreeOfLifeClassifier'] = TreeOfLifeClassifier
        g['CustomLabelsClassifier'] = CustomLabelsClassifier
        g['CustomLabelsBinningClassifier'] = CustomLabelsBinningClassifier
        return g[name]
    raise AttributeError(f"module 'bioclip' has no attribute {name!r}")
