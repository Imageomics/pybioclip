# pybioclip


[![PyPI - Version](https://img.shields.io/pypi/v/bioclip.svg)](https://pypi.org/project/bioclip)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bioclip.svg)](https://pypi.org/project/bioclip)

-----

Command line tool and python package to simplify using [BioCLIP](https://imageomics.github.io/bioclip/).


**Table of Contents**

- [Installation](#installation)
- [Python Package Usage](#python-package-usage)
- [Command Line Usage](#command-line-usage)
- [Acknowledgments](#acknowledgments)
- [License](#license)
  
## Requirements
- Python compatible with [PyTorch](https://pytorch.org/get-started/locally/#linux-python)

## Installation

```console
pip install git+https://github.com/Imageomics/pybioclip
```

If you have any issues with installation, please first upgrade pip by running `pip install --upgrade pip`.

## Python Package Usage
### Predict species classification

```python
from bioclip import TreeOfLifeClassifier, Rank

classifier = TreeOfLifeClassifier()
predictions = classifier.predict("Ursus-arctos.jpeg", Rank.SPECIES)

for prediction in predictions:
    print(prediction["species"], "-", prediction["score"])
```

Output:
```console
Ursus arctos - 0.9356034994125366
Ursus arctos syriacus - 0.05616999790072441
Ursus arctos bruinosus - 0.004126196261495352
Ursus arctus - 0.0024959812872111797
Ursus americanus - 0.0005009894957765937
```

Output from the `predict()` method showing the dictionary structure:
```
[{
    'kingdom': 'Animalia',
    'phylum': 'Chordata',
    'class': 'Mammalia',
    'order': 'Carnivora',
    'family': 'Ursidae',
    'genus': 'Ursus',
    'species_epithet': 'arctos',
    'species': 'Ursus arctos',
    'common_name': 'Kodiak bear'
    'score': 0.9356034994125366
}]
```

The output from the predict function can be converted into a [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) like so:
```python
import pandas as pd
from bioclip import TreeOfLifeClassifier, Rank

classifier = TreeOfLifeClassifier()
predictions = classifier.predict("Ursus-arctos.jpeg", Rank.SPECIES)
df = pd.DataFrame(predictions)
```

### Predict from a list of classes
```python
from bioclip import CustomLabelsClassifier

classifier = CustomLabelsClassifier()
predictions = classifier.predict("Ursus-arctos.jpeg", ["duck","fish","bear"])
for prediction in predictions:
   print(prediction["classification"], prediction["score"])
```
Output:
```console
duck 1.0306726583309e-09
fish 2.932403668845507e-12
bear 1.0
```

## Command Line Usage
```
bioclip predict [options] [IMAGE_FILE...]

Arguments:
  IMAGE_FILE         input image file

Options:
  -h --help
  --format=FORMAT    format of the output (table or csv) [default: csv]
  --rank=RANK        rank of the classification (kingdom, phylum, class, order, family, genus, species)
                     [default: species] 
  --k=K              number of top predictions to show [default: 5]
  --cls=CLS          comma separated list of classes to predict, when specified the --rank and
                     --k arguments are ignored [default: all]
  --output=OUTFILE   print output to file OUTFILE [default: stdout]
```

### Predict classification

#### Predict species for an image
The example images used below are [`Ursus-arctos.jpeg`](https://huggingface.co/spaces/imageomics/bioclip-demo/blob/ef075807a55687b320427196ac1662b9383f988f/examples/Ursus-arctos.jpeg) 
and [`Felis-catus.jpeg`](https://huggingface.co/spaces/imageomics/bioclip-demo/blob/ef075807a55687b320427196ac1662b9383f988f/examples/Felis-catus.jpeg) both from the [bioclip-demo](https://huggingface.co/spaces/imageomics/bioclip-demo).

Predict species for an `Ursus-arctos.jpeg` file:
```console
bioclip predict Ursus-arctos.jpeg
```
Output:
```
bioclip predict Ursus-arctos.jpeg
file_name,kingdom,phylum,class,order,family,genus,species_epithet,species,common_name,score
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctos,Ursus arctos,Kodiak bear,0.9356034994125366
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctos syriacus,Ursus arctos syriacus,syrian brown bear,0.05616999790072441
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctos bruinosus,Ursus arctos bruinosus,,0.004126196261495352
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctus,Ursus arctus,,0.0024959812872111797
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,americanus,Ursus americanus,Louisiana black bear,0.0005009894957765937
```

#### Predict species for multiple images saving to a file

To make predictions for files `Ursus-arctos.jpeg` and `Felis-catus.jpeg` saving the output to a file named `predictions.csv`:
```console
bioclip predict --output predictions.csv Ursus-arctos.jpeg Felis-catus.jpeg
```
The contents of `predictions.csv` will look like this: 
```
file_name,kingdom,phylum,class,order,family,genus,species_epithet,species,common_name,score
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctos,Ursus arctos,Kodiak bear,0.9356034994125366
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctos syriacus,Ursus arctos syriacus,syrian brown bear,0.05616999790072441
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctos bruinosus,Ursus arctos bruinosus,,0.004126196261495352
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctus,Ursus arctus,,0.0024959812872111797
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,americanus,Ursus americanus,Louisiana black bear,0.0005009894957765937
Felis-catus.jpeg,Animalia,Chordata,Mammalia,Carnivora,Felidae,Felis,silvestris,Felis silvestris,European Wildcat,0.7221033573150635
Felis-catus.jpeg,Animalia,Chordata,Mammalia,Carnivora,Felidae,Felis,catus,Felis catus,Domestic Cat,0.19810837507247925
Felis-catus.jpeg,Animalia,Chordata,Mammalia,Carnivora,Felidae,Felis,margarita,Felis margarita,Sand Cat,0.02798456884920597
Felis-catus.jpeg,Animalia,Chordata,Mammalia,Carnivora,Felidae,Lynx,felis,Lynx felis,,0.021829601377248764
Felis-catus.jpeg,Animalia,Chordata,Mammalia,Carnivora,Felidae,Felis,bieti,Felis bieti,Chinese desert cat,0.010979168117046356
```

#### Predict top 3 genera for an image and display output as a table
```console
bioclip predict --format table --k 3 --rank=genus Ursus-arctos.jpeg
```

Output:
```
+-------------------+----------+----------+----------+--------------+----------+--------+------------------------+
|     file_name     | kingdom  |  phylum  |  class   |    order     |  family  | genus  |         score          |
+-------------------+----------+----------+----------+--------------+----------+--------+------------------------+
| Ursus-arctos.jpeg | Animalia | Chordata | Mammalia |  Carnivora   | Ursidae  | Ursus  |   0.9994320273399353   |
| Ursus-arctos.jpeg | Animalia | Chordata | Mammalia | Artiodactyla | Cervidae | Cervus | 0.00032594642834737897 |
| Ursus-arctos.jpeg | Animalia | Chordata | Mammalia | Artiodactyla | Cervidae | Alces  | 7.803700282238424e-05  |
+-------------------+----------+----------+----------+--------------+----------+--------+------------------------+
```

### Predict from a list of classes
Create predictions for 3 classes (cat, bird, and bear) for image `Ursus-arctos.jpeg`:
```console
bioclip predict --cls cat,bird,bear Ursus-arctos.jpeg
```
Output:
```
file_name,classification,score
Ursus-arctos.jpeg,cat,4.581644930112816e-08
Ursus-arctos.jpeg,bird,3.051998476166773e-08
Ursus-arctos.jpeg,bear,0.9999998807907104                                                                 
```

### View command line help
```console
bioclip --help
```

## License

`pybioclip` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Acknowledgments
The [prediction code in this repo](src/bioclip/predict.py) is based on work by [@samuelstevens](https://github.com/samuelstevens) in [bioclip-demo](https://huggingface.co/spaces/imageomics/bioclip-demo/tree/ef075807a55687b320427196ac1662b9383f988f).
