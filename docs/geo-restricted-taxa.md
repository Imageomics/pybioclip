# Improving Prediction Performance

## Prediction on Geographically Restricted List

`pybioclip` provides three features for making predictions using a restricted set of terms:

- `--cls`
- `--bins`
- `--subset`

One way to populate the restricted set of terms is to use taxa that are known to have been observed in a specific geographic region.

There are many ways to abtain a list with this constraint. Two noteworthy methods are:

- Using the [Global Biodiversity Information Facility (GBIF)](https://www.gbif.org/) to download a list of taxa that have been observed in a specific geographic region.
- Using [Map of Life](https://mol.org/) to download a list of taxa that have been observed in a specific geographic region.

In our example, we will use the image below of the nēnē, or the Hawaiian goose (Branta sandvicensis). Its full taxonomic classification is Animalia Chordata Aves Anseriformes Anatidae Branta sandvicensis. 

**Save this image to your working directory as `nene.jpg` for the example.**

![Image title](https://inaturalist-open-data.s3.amazonaws.com/photos/13264181/medium.jpg){ loading=lazy }
/// caption
[© Kevin Schafer](https://www.inaturalist.org/photos/13264181)
///


## GBIF Web Interface Tutorial

The [Global Biodiversity Information Facility (GBIF)](https://www.gbif.org/) provides a web interface that enables users to produce a species list filtered by geographic coordinates. A brief tutorial is provided.

### Prerequisites
You must have an account with GBIF to create downloads with a DOI. You do not need an account to download data with a DOI created by others.

Visit [https://www.gbif.org/](https://www.gbif.org/) and click `Login` and `Register` to create an account.

Once you have an account, log in to manage data downloads from your profile.

### Getting a List of Taxa in a Geographic Region

!!! example "Example: filter for all occurrences of birds in the islands of Hawai'i."

    !!! note 
        Rather than creating a new download, you may use [https://doi.org/10.15468/dl.469rtz](https://doi.org/10.15468/dl.469rtz), which was prepared for this example using the steps below. Taxa have been extracted from the species list in this download and are available in:

        - [hawaii_bird_species_list.txt](assets/hawaii_bird_species_list.txt)
        - [hawaii_bird_families_list.txt](assets/hawaii_bird_families_list.txt)
        - [hawaii_bird_family_bins_list.csv](assets/hawaii_bird_family_bins_list.csv) (terms for bins of families were created manually)
        
        You can save these files to your working directory for the example.

    1. Log in to [https://www.gbif.org/](https://www.gbif.org/) and click `OCCURRENCES` near the search bar.
    2. Apply filters for your search (you can apply the filters in any combination and order).
        - To filter by region, use `Location` with `Including coordinates`, `Administrative areas (gadm.org)`, `Country or area`, or `Continent`.
        - The `Location` feature is very flexible. You may use the built-in map widget to specify one or more regions with polygons or squares by hand. You may also specify one or more square ranges by entering precise coordinates into `Range` and/or `Geometry`. These features are able to be used in combination.
        - Under the `Scientific name` filter, type "Aves", and select "Aves Class" from the search results.
        - Under the `Location` filter, click `Geometry`, and enter the following into the "WKT or GeoJSON" field:
        ```
        POLYGON((-160.54 18.54, -154.46 18.54, -154.46 22.26, -160.54 22.26, -160.54 18.54))
        ```
        - Click `ADD`
    3. Download the data.
        - Once the query completes, click `DOWNLOAD`.
        - Select the `SPECIES LIST`, read and agree to the terms of use, and click `UNDERSTOOD`.
        - The download will be prepared and associated with the DOI displayed at the top of the page, and you will receive an email when it is ready. 
        - The CSV will contain a list of taxa that meet the requirements of the filter with contents as defined by the [GBIF "Species list" download format](https://techdocs.gbif.org/en/data-use/download-formats#species-list).
        - You may extract contents of the CSV to a text file as you see fit for use with `pybioclip`, as we have with `hawaii_bird_species_list.txt`, `hawaii_bird_families_list.txt`, and `hawaii_bird_family_bins_list.csv`.

### Using a List of Taxa in a Geographic Region for Prediction

!!! example "Example: Perform predictions using the list of birds in the islands of Hawai'i."

    - Predict the family using the geographically restricted list [hawaii_bird_families_list.txt](assets/hawaii_bird_families_list.txt) 
    ```console
    bioclip predict --k 3 --format table --cls hawaii_bird_families_list.txt nene.jpg
    +-----------+----------------+---------------------+
    | file_name | classification |        score        |
    +-----------+----------------+---------------------+
    |  nene.jpg |    Anatidae    | 0.34369930624961853 |
    |  nene.jpg |   Ciconiidae   | 0.28970828652381897 |
    |  nene.jpg |    Gruidae     | 0.17063161730766296 |
    +-----------+----------------+---------------------+
    ```
    - Predict the species with open ended classification.
    ```console
    bioclip predict --k 5 --format table nene.jpg
    +-----------+----------+----------+-------+--------------+----------+--------+-----------------+---------------------+----------------+---------------------+
    | file_name | kingdom  |  phylum  | class |    order     |  family  | genus  | species_epithet |       species       |  common_name   |        score        |
    +-----------+----------+----------+-------+--------------+----------+--------+-----------------+---------------------+----------------+---------------------+
    |  nene.jpg | Animalia | Chordata |  Aves |  Gruiformes  | Gruidae  |  Grus  |       grus      |      Grus grus      |  Common Crane  |  0.7501388788223267 |
    |  nene.jpg | Animalia | Chordata |  Aves |  Gruiformes  | Gruidae  |  Grus  |     communis    |    Grus communis    |                | 0.07198520749807358 |
    |  nene.jpg | Animalia | Chordata |  Aves | Anseriformes | Anatidae | Branta |   sandvicensis  | Branta sandvicensis | Hawaiian Goose | 0.04125870019197464 |
    |  nene.jpg | Animalia | Chordata |  Aves |  Gruiformes  | Gruidae  |  Grus  |     cinerea     |     Grus cinerea    |                | 0.03031359612941742 |
    |  nene.jpg | Animalia | Chordata |  Aves |  Gruiformes  | Gruidae  |  Grus  |     monacha     |     Grus monacha    |  Hooded Crane  | 0.02994249388575554 |
    +-----------+----------+----------+-------+--------------+----------+--------+-----------------+---------------------+----------------+---------------------+
    ```
    The top two predictions are incorrect. However, we can use BioCLIP's predictions alongside the geographically restricted list to make a more informed decision. We can imagine that all we know about the image is that it was taken on one of the Hawaiian islands.
    - Predict the species combining open-ended classification with the geographically restricted list [hawaii_bird_species_list.txt](assets/hawaii_bird_species_list.txt).
        - First, predict the species with open-ended classification, saving the predictions to a file.
        ```console
        bioclip predict --k 20 nene.jpg > bioclip_predictions.csv
        head -n 4 bioclip_predictions.csv
        file_name,kingdom,phylum,class,order,family,genus,species_epithet,species,common_name,score
        nene.jpg,Animalia,Chordata,Aves,Gruiformes,Gruidae,Grus,grus,Grus grus,Common Crane,0.7501388788223267
        nene.jpg,Animalia,Chordata,Aves,Gruiformes,Gruidae,Grus,communis,Grus communis,,0.07198520749807358
        nene.jpg,Animalia,Chordata,Aves,Anseriformes,Anatidae,Branta,sandvicensis,Branta sandvicensis,Hawaiian Goose,0.04125870019197464
        ```
        - Filter the predictions using the geographically restricted list. E.g. using Pandas in Python.
        ```python
        import pandas as pd

        # Load the open-ended predictions.
        df_predictions = pd.read_csv('bioclip_predictions.csv')

        # Load the geographically restricted list of species.
        df_hawaii = pd.read_csv('hawaii_bird_species_list.txt', header=None, names=['species'])
        
        # Filter the predictions to include only species in the geographically restricted list.
        df_filtered = df_predictions[df_predictions["species"].isin(df_hawaii["species"])]
        
        # Save the filtered list to a file.
        df_filtered["species"].to_csv("bioclip_predictions_filtered.txt", index=False, header=False)
        ```
        - Predict the species using the filtered list.
        ```console
        bioclip predict --k 3 --format table --cls bioclip_predictions_filtered.txt nene.jpg
        +-----------+---------------------+---------------------+
        | file_name |    classification   |        score        |
        +-----------+---------------------+---------------------+
        |  nene.jpg | Branta sandvicensis |  0.8363304734230042 |
        |  nene.jpg |    Rhea americana   | 0.10964243859052658 |
        |  nene.jpg |   Anser cygnoides   | 0.02564687840640545 |
        +-----------+---------------------+---------------------+
        ```
    - Predict directly using the geographically restricted list of species.
    
    !!! warning
        Using a large list of custom classes takes a significant amount of time to process, as embeddings must be calculated for each class.

    ```console
    bioclip predict --k 3 --format table --cls hawaii_bird_species_list.txt nene.jpg
    +-----------+---------------------+---------------------+
    | file_name |    classification   |        score        |
    +-----------+---------------------+---------------------+
    |  nene.jpg | Branta sandvicensis |  0.7710946798324585 |
    |  nene.jpg |    Rhea americana   | 0.10109006613492966 |
    |  nene.jpg |   Branta leucopsis  |  0.0416080504655838 |
    +-----------+---------------------+---------------------+
    ```

    - Predict directly using the binned list of families.
    ```console
    bioclip predict --k 3 --format table --bins hawaii_bird_family_bins_list.csv nene.jpg
    +-----------+-----------------------+---------------------+
    | file_name |     classification    |        score        |
    +-----------+-----------------------+---------------------+
    |  nene.jpg |       Waterfowl       |  0.3437104821205139 |
    |  nene.jpg | Shorebirds/Waterbirds |  0.3415525555610657 |
    |  nene.jpg |         Cranes        | 0.17063717544078827 |
    +-----------+-----------------------+---------------------+
    ```
