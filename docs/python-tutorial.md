# Python Tutorial
Before beginning this tutorial you need to [install pybioclip](index.md/#installation) and download two example images: [`Ursus-arctos.jpeg`](https://huggingface.co/spaces/imageomics/bioclip-demo/blob/ef075807a55687b320427196ac1662b9383f988f/examples/Ursus-arctos.jpeg) 
and [`Felis-catus.jpeg`](https://huggingface.co/spaces/imageomics/bioclip-demo/blob/ef075807a55687b320427196ac1662b9383f988f/examples/Felis-catus.jpeg).


## Predict species classification

```python
from bioclip import TreeOfLifeClassifier, Rank

classifier = TreeOfLifeClassifier()
predictions = classifier.predict("Ursus-arctos.jpeg", Rank.SPECIES)

for prediction in predictions:
    print(prediction["species"], "-", prediction["score"])
```

Output:
```
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

The first argument of the `predict()` method supports both a single path or a list of paths.

!!! info "Documentation"
    The [TreeOfLifeClassifier docs](python-api.md/#bioclip.TreeOfLifeClassifier) contains details about the arguments supported by the constructor and the `predict()` method.

## Predict from a list of classes
```python
from bioclip import CustomLabelsClassifier

classifier = CustomLabelsClassifier(["duck","fish","bear"])
predictions = classifier.predict("Ursus-arctos.jpeg")
for prediction in predictions:
   print(prediction["classification"], prediction["score"])
```
Output:
```
duck 1.0306726583309e-09
fish 2.932403668845507e-12
bear 1.0
```

!!! info "Documentation"
    The [CustomLabelsClassifier docs](python-api.md/#bioclip.CustomLabelsClassifier) contains details about the arguments supported by the constructor and the `predict()` method.

### Predict using a Custom Model
To predict with a custom model the `model_str` and `pretrained_str` arguments must be specified.
In this example the [CLIP-ViT-B-16-laion2B-s34B-b88K](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K) model is used.
```python
from bioclip import CustomLabelsClassifier

classifier = CustomLabelsClassifier(
    cls_ary = ["duck","fish","bear"],
    model_str='ViT-B-16',
    pretrained_str='laion2b_s34b_b88k')

print(classifier.predict("Ursus-arctos.jpeg"))
```

See [this tutorial](command-line-tutorial.md/#predict-using-a-custom-model) for instructions for listing available pretrained models.


## Predict from a list of classes with binning
```python
from bioclip import CustomLabelsBinningClassifier

classifier = CustomLabelsBinningClassifier(cls_to_bin={
  'dog': 'small',
  'fish': 'small',
  'bear': 'big',
})
predictions = classifier.predict("Ursus-arctos.jpeg")

for prediction in predictions:
   print(prediction["classification"], prediction["score"])
```
Output:
```
big 0.99992835521698
small 7.165559509303421e-05
```

!!! info "Documentation"
    The [CustomLabelsBinningClassifier documentation](python-api.md/#bioclip.CustomLabelsBinningClassifier) describes all arguments supported by the constructor. The base class [CustomLabelsClassifier docs](python-api.md/#bioclip.CustomLabelsClassifier) describes arguments for the predict method.

## Example Notebooks
### Predict species for images
[PredictImages.ipynb](https://github.com/Imageomics/pybioclip/blob/main/examples/PredictImages.ipynb)  downloads some images and predicts species.
<a target="_blank" href="https://colab.research.google.com/github/Imageomics/pybioclip/blob/main/examples/PredictImages.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Predict species for iNaturalist images
[iNaturalistPredict.ipynb](https://github.com/Imageomics/pybioclip/blob/main/examples/iNaturalistPredict.ipynb) downloads images from [inaturalist.org](https://www.inaturalist.org/) and predicts species.
<a target="_blank" href="https://colab.research.google.com/github/Imageomics/pybioclip/blob/main/examples/iNaturalistPredict.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Predict using a subset of the TreeOfLife
[TOL-Subsetting.ipynb](https://github.com/Imageomics/pybioclip/blob/main/examples/TOL-Subsetting.ipynb) filters the TreeOfLife embeddings.
<a target="_blank" href="https://colab.research.google.com/github/Imageomics/pybioclip/blob/main/examples/TOL-Subsetting.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

!!! info "Documentation"
     For subsetting the TreeOfLifeClassifier see [get_label_data()](python-api.md#bioclip.TreeOfLifeClassifier.get_label_data), [create_taxa_filter()](python-api.md#bioclip.TreeOfLifeClassifier.create_taxa_filter) and [apply_filter()](python-api.md#bioclip.TreeOfLifeClassifier.apply_filter) .

### Experiment with [grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
[GradCamExperiment.ipynb](https://github.com/Imageomics/pybioclip/blob/main/examples/GradCamExperiment.ipynb)  applies GradCAM AI explainability to BioCLIP. <a target="_blank" href="https://colab.research.google.com/github/Imageomics/pybioclip/blob/main/examples/GradCamExperiment.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Fine-tune
#### Notebooks
The following notebooks show methods to fine-tune BioCLIP for classification.

- [FineTuneSVM.ipynb](https://github.com/Imageomics/pybioclip/blob/main/examples/FineTuneSVM.ipynb) fine-tunes  BioCLIP by combining an [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) with BioCLIP image embeddings. <a target="_blank" href="https://colab.research.google.com/github/Imageomics/pybioclip/blob/main/examples/FineTuneSVM.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- [FineTuneRidgeClassifier.ipynb](https://github.com/Imageomics/pybioclip/blob/main/examples/FineTuneRidgeClassifier.ipynb)
fine-tunes BioCLIP by combining a [RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html) with BioCLIP image embeddings. <a target="_blank" href="https://colab.research.google.com/github/Imageomics/pybioclip/blob/main/examples/FineTuneRidgeClassifier.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- [FineTuneSimpleShot.ipynb](https://github.com/Imageomics/pybioclip/blob/main/examples/FineTuneSimpleShot.ipynb)
fine-tunes BioCLIP by combining a [SimpleShot](https://arxiv.org/abs/1911.04623) classifier with BioCLIP image embeddings. <a target="_blank" href="https://colab.research.google.com/github/Imageomics/pybioclip/blob/main/examples/FineTuneSimpleShot.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As can be seen from comparing the confusion matrices in the notebooks, fine-tuning may yield better results than using BioCLIP in "zero-shot mode", i.e., predicting on a list of custom labels.

This work is based on code from [biobench](https://github.com/samuelstevens/biobench).

#### Comparison of methods

| Method | Maximum Classes | Minimum Training Data |
|---|---|---|
| SVM | ~20 | 5+ examples |
| Ridge Classifier | No maximum | 10+ examples per class |
| SimpleShot | No maximum | 1+ example per class |

- **SVMs** can support linear and non-linear boundaries and are suitable for binary classification or fewer than ~20 classes (because you train a one-vs-rest for each class).
- **Ridge classifiers** are best for linear classification tasks. They require training but are powerful classifiers for many, many tasks, especially with sufficient data.
- **SimpleShot** is extremely data-efficient and works well for multiple (20+) classes.

## PIL Images
The predict() functions used in all the examples above allow passing a list of paths or a list of [PIL Images](https://pillow.readthedocs.io/en/stable/reference/Image.html).
When a list of PIL images is passed the index of the image will be filled in for `file_name`. This is because PIL images may not have an associated file name.
