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

### Predict classification

#### Example: Predict species for an image
The example image used below is [`Ursus-arctos.jpeg`](https://huggingface.co/spaces/imageomics/bioclip-demo/blob/ef075807a55687b320427196ac1662b9383f988f/examples/Ursus-arctos.jpeg) from the [bioclip-demo](https://huggingface.co/spaces/imageomics/bioclip-demo).

Predict species for an `Ursus-arctos.jpeg` file:
```console
bioclip predict Ursus-arctos.jpeg
```
Output:
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

---

To save as a CSV or JSON file you can use the `--format <file type>` and `--output <filename>` arguments with `csv` or `json`, respectively.

To save the JSON output to `ursus.json` run:
```console
bioclip predict --format json --output ursus.json Ursus-arctos.jpeg 
```

To save the CSV output to `ursus.csv` run:
```console
bioclip predict --format csv --output ursus.csv Ursus-arctos.jpeg 
```

#### Predict genus for an image

Predict genus for image `Ursus-arctos.jpeg`, restricted to the top 3 predictions:
```console
bioclip predict --rank genus --k 3 Ursus-arctos.jpeg
```
Output:
```
+---------------------------------------------------------+------------------------+
|                          Taxon                          |      Probability       |
+---------------------------------------------------------+------------------------+
|    Animalia Chordata Mammalia Carnivora Ursidae Ursus   |   0.9994320273399353   |
| Animalia Chordata Mammalia Artiodactyla Cervidae Cervus | 0.00032594642834737897 |
|  Animalia Chordata Mammalia Artiodactyla Cervidae Alces | 7.803700282238424e-05  |
+---------------------------------------------------------+------------------------+
```

#### Optional arguments for predicting classifications:
- `--rank RANK` - rank of the classification (kingdom, phylum, class, order, family, genus, species) [default: species] 
- `--k K` - number of top predictions to show [default: 5]
- `--format FORMAT` - format of the output (table, json, or csv) [default: table]
- `--output OUTPUT` - save output to a filename instead of printing it [default: stdout]


### Predict from a list of classes

Create predictions for 3 classes (cat, bird, and bear) for image `Ursus-arctos.jpeg`:
```console
bioclip predict --cls cat,bird,bear Ursus-arctos.jpeg
```
Output:
```
+-------+-----------------------+
| Taxon |      Probability      |
+-------+-----------------------+
|  cat  | 4.581644930112816e-08 |
|  bird | 3.051998476166773e-08 |
|  bear |   0.9999998807907104  |
+-------+-----------------------+%                                                                  
```

#### Optional arguments for predicting from a list of classes:
- `--format FORMAT` - format of the output (table, json, or csv) [default: table]
- `--output OUTPUT` - save output to a filename instead of printing it [default: stdout]
- `--cls CLS` - comma separated list of classes to predict, when specified the `--rank` and `--k` arguments are ignored [default: all]


### View command line help
```console
bioclip --help
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

`pybioclip` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Acknowledgments
The [prediction code in this repo](src/bioclip/predict.py) is based on work by [@samuelstevens](https://github.com/samuelstevens) in [bioclip-demo](https://huggingface.co/spaces/imageomics/bioclip-demo/tree/ef075807a55687b320427196ac1662b9383f988f).
