# pybioclip

Command line tool and python package to simplify the use of [BioCLIP](https://imageomics.github.io/bioclip/).

Key features include:

- [**Taxonomic label prediction**](command-line-tutorial.md#predict-species-for-an-image) for images across ranks in the Linnaean hierarchy (tunable from kingdom to species).
- [**Custom label predictions**](command-line-tutorial.md#predict-from-a-list-of-classes) from user-supplied classification categories.
- [**Image embedding generation**](command-line-tutorial.md#create-embedding-for-an-image) in a text-aligned feature space.
- [**Batch image processing**](command-line-tutorial.md#predict-species-for-multiple-images-saving-to-a-file) with performance optimizations.
- [**Containers provided**](docker.md) to simplfy use in computational pipelines.

No particular coding knowledge of ML or computer vision is required to use pybioclip.

## Installation
Requires python that is compatible with [PyTorch](https://pytorch.org/get-started/locally/#linux-python).

```console
pip install pybioclip
```
If you have any issues with installation, please first upgrade pip by running `pip install --upgrade pip`.


## Tutorials
[Command Line Tutorial](command-line-tutorial.md){ .md-button .md-button--primary } [Python Tutorial](python-tutorial.md){ .md-button .md-button--primary }
