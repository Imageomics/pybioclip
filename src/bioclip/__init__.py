# SPDX-FileCopyrightText: 2024-present John Bradley <johnbradley2008@gmail.com>
#
# SPDX-License-Identifier: MIT

__all__ = ["TreeOfLifeClassifier", "Rank", "CustomLabelsClassifier", "CustomLabelsBinningClassifier"]

def __getattr__(name):
    if name in __all__:
        from bioclip.predict import (
            TreeOfLifeClassifier,
            Rank,
            CustomLabelsClassifier,
            CustomLabelsBinningClassifier,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")
