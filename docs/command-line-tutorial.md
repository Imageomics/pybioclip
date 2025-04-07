# Command Line Tutorial

Before beginning this tutorial you need to [install pybioclip](index.md#installation) and download two example images: [`Ursus-arctos.jpeg`](https://huggingface.co/spaces/imageomics/bioclip-demo/blob/ef075807a55687b320427196ac1662b9383f988f/examples/Ursus-arctos.jpeg)
and [`Felis-catus.jpeg`](https://huggingface.co/spaces/imageomics/bioclip-demo/blob/ef075807a55687b320427196ac1662b9383f988f/examples/Felis-catus.jpeg).

## Tree Of Life Predictions
The `bioclip predict` command, when not supplying a custom list of labels, will create a prediction based on the [BioCLIP tree of life embeddings](https://huggingface.co/spaces/imageomics/bioclip-demo/blob/main/txt_emb_species.npy).

### Predict species for an image

Predict species for an `Ursus-arctos.jpeg` file:
```console
bioclip predict Ursus-arctos.jpeg
```
Output:
```
file_name,kingdom,phylum,class,order,family,genus,species_epithet,species,common_name,score
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctos,Ursus arctos,Kodiak bear,0.9356034994125366
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctos syriacus,Ursus arctos syriacus,syrian brown bear,0.05616999790072441
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctos bruinosus,Ursus arctos bruinosus,,0.004126196261495352
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,arctus,Ursus arctus,,0.0024959812872111797
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Carnivora,Ursidae,Ursus,americanus,Ursus americanus,Louisiana black bear,0.0005009894957765937
```
!!! info "Documentation"
    The [bioclip predict documentation](command-line-help.md/#bioclip-predict) describes all arguments supported by `bioclip predict` command.

### Predict species for multiple images saving to a file

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
!!! info "Documentation"
    The [bioclip predict documentation](command-line-help.md/#bioclip-predict) describes all arguments supported by `bioclip predict` command.

### Predict top 3 genera for an image and display output as a table
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

!!! info "Documentation"
    The [bioclip predict documentation](command-line-help.md/#bioclip-predict) describes all arguments supported by `bioclip predict` command.

### Predict using a TOL subset
The `predict` command has support for a `--subset <csv-path>` argument. The first column in the CSV file must be named kingdom, phylum, class, order, family, genus, or species. The values must match the TOL labels otherwise an error occurs. See the [bioclip list-tol-taxa](command-line-help.md/#bioclip-list-tol-taxa) command to create a CSV of TOL labels.

In this example we create a CSV to subset TOL to two orders.
Create a CSV named `orders.csv` with the following content:
```
order
Artiodactyla
Rodentia
```

```console
bioclip predict --subset orders.csv Ursus-arctos.jpeg
```

Output:
```
file_name,kingdom,phylum,class,order,family,genus,species_epithet,species,common_name,score
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Artiodactyla,Cervidae,Cervus,canadensis sibericus,Cervus canadensis sibericus,,0.7347981333732605
Ursus-arctos.jpeg,Animalia,Chordata,Mammalia,Artiodactyla,Cervidae,Alces,alces,Alces alces,European elk,0.17302176356315613
...
```

!!! info "Documentation"
    The [bioclip predict documentation](command-line-help.md/#bioclip-predict) describes all arguments supported by `bioclip predict` command.



## Custom Label Predictions
To predict with custom labels using the `bioclip predict` command the `--cls` or `--bins` arguments must be used.

### Predict from a list of classes
Create predictions for 3 classes (cat, bird, and bear) for image `Ursus-arctos.jpeg`:
```console
bioclip predict --cls cat,bird,bear Ursus-arctos.jpeg
```
Output:
```
file_name,classification,score
Ursus-arctos.jpeg,bear,0.9999998807907104
Ursus-arctos.jpeg,cat,4.581697155003894e-08
Ursus-arctos.jpeg,bird,3.052039332374079e-08
```
!!! info "Documentation"
    The [bioclip predict documentation](command-line-help.md/#bioclip-predict) describes all arguments supported by `bioclip predict` command.

### Predict from a binning CSV
Create predictions for 3 classes (cat, bird, and bear) with 2 bins (one, two) for image `Ursus-arctos.jpeg`:

Create a CSV file named `bins.csv` with the following contents:
```
cls,bin
cat,one
bird,one
bear,two
```
The names of the columns do not matter. The first column values will be used as the classes. The second column values will be used for bin names.

Run predict command:
```console
bioclip predict --bins bins.csv Ursus-arctos.jpeg
```

Output:
```
Ursus-arctos.jpeg,two,0.9999998807907104
Ursus-arctos.jpeg,one,7.633736487377973e-08
```
!!! info "Documentation"
    The [bioclip predict documentation](command-line-help.md/#bioclip-predict) describes all arguments supported by `bioclip predict` command.

### Predict using a custom model
List available models:
```console
bioclip list-models
```
Output:
```
...
ViT-B-16
ViT-B-16-plus
...
```
List pretrained models for a model:
```console
bioclip list-models --model  ViT-B-16
```
Output:
```
...
laion2b_s34b_b88k
...
```
Create a prediction:
```console
bioclip predict --cls duck,fish,bear --model ViT-B-16 --pretrained laion2b_s34b_b88k Ursus-arctos.jpeg
```
Output:
```
file_name,classification,score
Ursus-arctos.jpeg,bear,0.9999988079071045
Ursus-arctos.jpeg,fish,1.1098603636128246e-06
Ursus-arctos.jpeg,duck,2.7479762465532076e-08
```

## Create embeddings
The `bioclip embed` command creates an embedding for one or more image files.

### Create embedding for an image
```console
bioclip embed Ursus-arctos.jpeg
```
Output:
```
{
    "model": "hf-hub:imageomics/bioclip",
    "embeddings": {
        "Ursus-arctos.jpeg": [
            -0.23633578419685364,
            -0.28467196226119995,
            -0.4394485652446747,
            ...
        ]
    }
}
```
!!! info "Documentation"
    The [bioclip embed documentation](command-line-help.md/#bioclip-embed) describes all arguments supported by `bioclip embed` command.

## View command line help
```console
bioclip -h
bioclip <command> -h
```
