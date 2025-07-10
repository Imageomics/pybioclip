# pybioclip


[![PyPI - Version](https://img.shields.io/pypi/v/pybioclip.svg)](https://pypi.org/project/pybioclip)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pybioclip.svg)](https://pypi.org/project/pybioclip)

-----

Command line tool and python package to simplify using [BioCLIP](https://imageomics.github.io/bioclip/), including for taxonomic or other label prediction on (and thus annotation or labeling of) images, as well as for generating semantic embeddings for images. No particular understanding of ML or computer vision is required to use it. It also implements a number of performance optimizations for batches of images or custom class lists, which should be particularly useful for integration into computational workflows.

## Documentation
See the [pybioclip documentation website](https://imageomics.github.io/pybioclip/) for requirements, installation instructions, and tutorials.

## License

`pybioclip` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Citation

Our code (this repository):
```
@software{Bradley_pybioclip_2025,
author = {Bradley, John and Lapp, Hilmar and Campolongo, Elizabeth G.},
doi = {10.5281/zenodo.13151194},
month = jul,
title = {{pybioclip}},
version = {2.0.0},
year = {2025}
}
```

BioCLIP paper:
```
@inproceedings{stevens2024bioclip,
  title = {{B}io{CLIP}: A Vision Foundation Model for the Tree of Life}, 
  author = {Samuel Stevens and Jiaman Wu and Matthew J Thompson and Elizabeth G Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024}
}
```

Also consider citing the BioCLIP code:
```
@software{bioclip2023code,
  author = {Samuel Stevens and Jiaman Wu and Matthew J. Thompson and Elizabeth G. Campolongo and Chan Hee Song and David Edward Carlyn},
  doi = {10.5281/zenodo.10895871},
  title = {BioCLIP},
  version = {v1.0.0},
  year = {2024}
}
```
