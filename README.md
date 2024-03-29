# pybioclip


[![PyPI - Version](https://img.shields.io/pypi/v/bioclip.svg)](https://pypi.org/project/bioclip)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bioclip.svg)](https://pypi.org/project/bioclip)

-----

Command line tool and python package to simplify using [BioCLIP](https://imageomics.github.io/bioclip/).


**Table of Contents**

- [Installation](#installation)
- [Command Line Usage](#command-line-usage)
- [Python Package Usage](#python-package-usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [License](#license)
  
## Requirements
- Python compatible with [PyTorch](https://pytorch.org/get-started/locally/#linux-python)

## Installation

```console
pip install git+https://github.com/Imageomics/pybioclip
```

If you have any issues with installation, please first upgrade pip by running `pip install --upgrade pip`.

## Command Line Usage

### Viewing command line help
```console
$ bioclip --help
```

### Predict classification

Example: Predict taxon for image "Ursus-arctos.jpeg":
```console
bioclip predict Ursus-arctos.jpeg
```

```
+----------------------------------------------------------------------------------------+-----------------------+
|                                         Taxon                                          |      Probability      |
+----------------------------------------------------------------------------------------+-----------------------+
|        Animalia Chordata Mammalia Carnivora Ursidae Ursus arctos (Kodiak bear)         |   0.9356034994125366  |
| Animalia Chordata Mammalia Carnivora Ursidae Ursus arctos syriacus (syrian brown bear) |  0.05616999790072441  |
|          Animalia Chordata Mammalia Carnivora Ursidae Ursus arctos bruinosus           |  0.004126196261495352 |
|               Animalia Chordata Mammalia Carnivora Ursidae Ursus arctus                | 0.0024959812872111797 |
|  Animalia Chordata Mammalia Carnivora Ursidae Ursus americanus (Louisiana black bear)  | 0.0005009894957765937 |
+----------------------------------------------------------------------------------------+-----------------------+
```

Usage for prediction
```
bioclip predict [--format=FORMAT --rank=RANK --k=K --output=OUTFILE] IMAGE_FILE 
```


### Predict from a list of classes
```
bioclip predict [--format=FORMAT --cls=CLS --output=OUTFILE] IMAGE_FILE 
```

## Python Package Usage
### Predict species classification

```python
from bioclip import predict_classification, Rank

predictions = predict_classification("Ursus-arctos.jpeg", Rank.SPECIES)

for species_name, probability in predictions.items():
   print(species_name, probability)
```

Output:
```console
Animalia Chordata Mammalia Carnivora Ursidae Ursus arctos (Kodiak bear) 0.9356034994125366
Animalia Chordata Mammalia Carnivora Ursidae Ursus arctos syriacus (syrian brown bear) 0.05616999790072441
Animalia Chordata Mammalia Carnivora Ursidae Ursus arctos bruinosus 0.004126196261495352
Animalia Chordata Mammalia Carnivora Ursidae Ursus arctus 0.0024959812872111797
Animalia Chordata Mammalia Carnivora Ursidae Ursus americanus (Louisiana black bear) 0.0005009894957765937
```

### Predict from a list of classes
```python
from bioclip import predict_classifications_from_list, Rank

predictions = predict_classifications_from_list("Ursus-arctos.jpeg",
                                                ["duck","fish","bear"])

for cls, probability in predictions.items():
   print(cls, probability)
```
Output:
```console
duck 1.0306726583309e-09
fish 2.932403668845507e-12
bear 1.0
```

## License

`bioclip` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Acknowledgments
The [prediction code in this repo](src/bioclip/predict.py) is based on work by [@samuelstevens](https://github.com/samuelstevens) in [bioclip-demo](https://huggingface.co/spaces/imageomics/bioclip-demo/tree/ef075807a55687b320427196ac1662b9383f988f)
