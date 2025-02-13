# Improving Prediction Performance

## Prediction on Geographically Restricted List

`pybioclip` provides three features for making predictions using a restricted set of terms:

- `--cls`
- `--bins`
- `--subset`

One way to populate the restricted set of terms is to use taxa that are known to have been observed in a specific geographic region.

There are many ways to obtain a list with this constraint. Two noteworthy methods are:

- Using the [Global Biodiversity Information Facility (GBIF)](https://www.gbif.org/) to download a list of taxa that have been observed in a specific geographic region.
- Using [Map of Life (MOL)](https://mol.org/) to download a list of taxa that have been observed in a specific geographic region.

In our examples, we will use the image below of the nēnē, or the Hawaiian goose (*Branta sandvicensis*). Its full taxonomic classification is *Animalia Chordata Aves Anseriformes Anatidae Branta sandvicensis*. 

**Save this image to your working directory as `nene.jpg` for the example.**

![picture of a nene](https://inaturalist-open-data.s3.amazonaws.com/photos/13264181/medium.jpg){ loading=lazy }
/// caption
[© Kevin Schafer](https://www.inaturalist.org/photos/13264181)
///


### Getting a List of Taxa in a Geographic Region

Both [GBIF](https://www.gbif.org/) and [MOL](https://mol.org) provide web interfaces that enable users to produce a species list filtered by geographic coordinates and regions. A brief tutorial is provided. These examples illustrate a small subset of the capabilities provided by these platforms.

#### GBIF Web Interface Tutorial

You must have an account with GBIF to create downloads with a DOI. You do not need an account to download data with a DOI created by others.

Visit [https://www.gbif.org/](https://www.gbif.org/) and click `Login` and `Register` to create an account.

Once you have an account, log in to manage data downloads from your profile.

!!! example "Example: filter for all occurrences of birds in the islands of Hawai'i with GBIF."

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
        - Under the `Location` filter, click `Geometry`, and enter the following into the "WKT or GeoJSON" field to specify the main eight islands of Hawai'i:
        ```
        POLYGON((-160.54 18.54, -154.46 18.54, -154.46 22.26, -160.54 22.26, -160.54 18.54))
        ```
            - You could also paste the equivalent raw content of the file [hawaii.geojson](assets/hawaii.geojson) into the "WKT or GeoJSON" field under `Geometry`.
        - Click `ADD`
    3. Download the data.
        - Once the query completes, click `DOWNLOAD`.
        - Select the `SPECIES LIST`, read and agree to the terms of use, and click `UNDERSTOOD`.
        - The download will be prepared and associated with the DOI displayed at the top of the page, and you will receive an email when it is ready. 
        - The CSV will contain a list of taxa that meet the requirements of the filter with contents as defined by the [GBIF "Species list" download format](https://techdocs.gbif.org/en/data-use/download-formats#species-list).
        - You may extract contents of the CSV to a text file as you see fit for use with `pybioclip`, as we have with `hawaii_bird_species_list.txt`, `hawaii_bird_families_list.txt`, and `hawaii_bird_family_bins_list.csv`.

#### MOL Web Interface Tutorial
There are no prerequisites. An account with MOL is optional and not required for downloading data.

!!! example "Example: filter for all species of birds in the islands of Hawai'i with MOL."
    1. Visit [https://mol.org/](https://mol.org/) and click `Regions`.
    2. Under "Search for an area by name:", type "Hawai'i" and select "Hawai'i, United States".
    3. Click `Go to Species Report`. This should take you to a page matching [this URL](https://mol.org/regions/region/species?regiontype=region&region_id=f470c3d3-03d2-4146-be2c-1b2f435a6eae).
    4. Under "Species", click `<N> Birds`, where `<N>` is the number of bird species in the region. This should append `&group=birds` to the URL.
    5. Refresh the browser to apply the Birds filter to the download. The filter will be applied when loading [the URL](https://mol.org/regions/region/species?regiontype=region&region_id=f470c3d3-03d2-4146-be2c-1b2f435a6eae&group=birds) with the Birds filter applied. Note that if you download before refreshing, the download will include all species in the region.
    6. Click `Download`, review the information in the pop-up about the download contents, and click `Download Now`.

### Using a List of Taxa in a Geographic Region for Prediction

These examples use the files containing information extracted from the GBIF download. You are encouraged to practice by creating similarly formatted files with the `scientific_name` and `family` columns from the `SpeciesList.csv` file in the MOL download.

!!! example "Example: Predict the species of an image using [open-ended classification](https://imageomics.github.io/pybioclip/command-line-tutorial/#predict-species-for-an-image)."
    ```console
    bioclip predict --k 5 --format table nene.jpg
    ```
    Output:
    ```console
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
    !!! warning "Incorrect top predictions."
        The top two predictions are incorrect. This suggests that filtering the possibilities for predicted labels may improve the accuracy of the predictions.

We can imagine that all we know about the image is that it was taken on one of the Hawaiian islands. We can use multiple approaches incorporating the geographically restricted lists to make a more informed decision. 

!!! example "Example: Predict the species of an image using the geographically restricted list of species directly as [custom labels](https://imageomics.github.io/pybioclip/command-line-tutorial/#custom-label-predictions)."
    Using [hawaii_bird_species_list.txt](assets/hawaii_bird_species_list.txt):
    ```console
    bioclip predict --k 3 --format table --cls hawaii_bird_species_list.txt nene.jpg
    ```
    Output:
    ```console
    +-----------+---------------------+---------------------+
    | file_name |    classification   |        score        |
    +-----------+---------------------+---------------------+
    |  nene.jpg | Branta sandvicensis |  0.7710946798324585 |
    |  nene.jpg |    Rhea americana   | 0.10109006613492966 |
    |  nene.jpg |   Branta leucopsis  |  0.0416080504655838 |
    +-----------+---------------------+---------------------+
    ```
    Advantages of this approach:

    - The predictions are more accurate because they are limited to species that are known to be present in the geographic region.
    
    Disadvantages of this approach:

    - The prediction is slow because the embeddings must be calculated for all species in the list. See the following examples for approaches to speed up prediction.
    - The predictions are limited to the species in the list, which may not include the correct species if it is an anomolous occurrence.

!!! example "Example: Predict the species of an image by combining the geographically restricted list with the precomputed embeddings of the [TOL subset](https://imageomics.github.io/pybioclip//command-line-tutorial/#predict-using-a-tol-subset) of taxa."
        
    Obtain the TOL subset.
    ```console
    bioclip list-tol-taxa > tol_subset.csv
    ```
    
    Filter the TOL subset to include only the geographically restricted list in [hawaii_bird_species_list.txt](assets/hawaii_bird_species_list.txt). For example, with Pandas in Python:
    ```python
    import pandas as pd

    # Load the TOL subset.
    df_tol = pd.read_csv('tol_subset.csv')

    # Load the list of species.
    df_species = pd.read_csv('hawaii_bird_species_list.txt', header=None, names=['species'])

    # Merge the TOL subset with the list of species, inner merge
    df_merged = pd.merge(df_tol, df_species, on='species')

    # Save the merged list to a file.
    df_merged["species"].to_csv("tol_subset_filtered.csv", index=False, header=True)
    ```

    Predict using the filtered TOL subset.
    ```console
    bioclip predict --k 3 --format table --subset tol_subset_filtered.csv nene.jpg
    ```
    Output:
    ```console
    +-----------+----------+----------+-------+--------------+----------+--------------+-----------------+---------------------+------------------+---------------------+
    | file_name | kingdom  |  phylum  | class |    order     |  family  |    genus     | species_epithet |       species       |   common_name    |       
    score        |
    +-----------+----------+----------+-------+--------------+----------+--------------+-----------------+---------------------+------------------+---------------------+
    |  nene.jpg | Animalia | Chordata |  Aves | Anseriformes | Anatidae |    Branta    |   sandvicensis  | Branta sandvicensis |  Hawaiian Goose  |  0.7040964365005493 |
    |  nene.jpg | Animalia | Chordata |  Aves |  Rheiformes  | Rheidae  |     Rhea     |    americana    |    Rhea americana   |   Greater Rhea   | 0.18546675145626068 |
    |  nene.jpg | Animalia | Chordata |  Aves |  Gruiformes  | Gruidae  | Anthropoides |      virgo      |  Anthropoides virgo | Demoiselle Crane | 0.04178838059306145 |
    +-----------+----------+----------+-------+--------------+----------+--------------+-----------------+---------------------+------------------+---------------------+
    ```

    Advantages of this approach:
    
    - Significant speedup in prediction time because the embeddings are precomputed.
    - The predictions are more accurate than open-ended prediction alone because they are limited to species that are known to be present in the geographic region.
    - In the case of an anomalous occurrence, we have BioCLIP's first best guess handy that is not constrained by the geographical list.
    - Using higher taxa in the subset list: If you know that your image or image set is from one family of organisms, or if you know it belongs to a certain order but some other order has species that look confusingly similar, then you may be well served using a higher-level taxon.

    Disadvantages of this approach:

    - Only taxa in the TOL subset can be included in the geographically restricted list, which may exclude entries present in the external list.
    - Using higher taxa in the subset list: The predictions are provided to the species level, even if the geographically restricted subset file lists only higher taxa. For example, if you want to remove taxa that don't live in your geographic area from prediction, and higher taxa include species that do and do not live in the area, then the species in the higher taxa will be included in the prediction. In this case, it's not advisable to use higher taxa in the geographically restricted list.


!!! example "Example: Predict the species of an image combining open-ended classification with the geographically restricted list using [custom labels](https://imageomics.github.io/pybioclip/command-line-tutorial/#predict-from-a-list-of-classes)."
    
    Predict the top 20 species with open-ended classification, saving the predictions to a file.
    ```console
    bioclip predict --k 20 nene.jpg > bioclip_predictions.csv
    ```
    Filter the predictions using the geographically restricted list of species in [hawaii_bird_species_list.txt](assets/hawaii_bird_species_list.txt). E.g. using Pandas in Python.
    ```python
    import pandas as pd

    # Load the open-ended predictions.
    df_predictions = pd.read_csv('bioclip_predictions.csv')

    # Load the geographically restricted list of species.
    df_hawaii_species = pd.read_csv('hawaii_bird_species_list.txt', header=None, names=['species'])
    
    # Filter the predictions to include only species in the geographically restricted list.
    df_merged = pd.merge(df_predictions, df_hawaii_species, on='species', how='inner')
    
    # Save the filtered list to a file.
    df_merged["species"].to_csv("bioclip_species_predictions_filtered.txt", index=False, header=False)
    ```
    In this example, this step removes the option to predict the crane species *Grus grus* and *Grus communis*, which are not found in Hawai'i.

    Predict the species using the filtered list.
    ```console
    bioclip predict --k 3 --format table --cls bioclip_species_predictions_filtered.txt nene.jpg
    ```
    Output:
    ```console
    +-----------+---------------------+---------------------+
    | file_name |    classification   |        score        |
    +-----------+---------------------+---------------------+
    |  nene.jpg | Branta sandvicensis |  0.8363304734230042 |
    |  nene.jpg |    Rhea americana   | 0.10964243859052658 |
    |  nene.jpg |   Anser cygnoides   | 0.02564687840640545 |
    +-----------+---------------------+---------------------+
    ```

!!! example "Example: Predict the family of an image among bird families in Hawai'i as custom classes."
    Using [hawaii_bird_families_list.txt](assets/hawaii_bird_families_list.txt):
    ```console
    bioclip predict --k 3 --format table --cls hawaii_bird_families_list.txt nene.jpg
    ```
    Output:
    ```console
    +-----------+----------------+---------------------+
    | file_name | classification |        score        |
    +-----------+----------------+---------------------+
    |  nene.jpg |    Anatidae    | 0.34369930624961853 |
    |  nene.jpg |   Ciconiidae   | 0.28970828652381897 |
    |  nene.jpg |    Gruidae     | 0.17063161730766296 |
    +-----------+----------------+---------------------+
    ```

!!! example "Example: Predict a custom [binned classification](https://imageomics.github.io/pybioclip/command-line-tutorial/#predict-from-a-binning-csv) of an image using the binned list of families."
    Using [hawaii_bird_family_bins_list.csv](assets/hawaii_bird_family_bins_list.csv):
    ```console
    bioclip predict --k 3 --format table --bins hawaii_bird_family_bins_list.csv nene.jpg
    ```
    Output:
    ```console
    +-----------+-----------------------+---------------------+
    | file_name |     classification    |        score        |
    +-----------+-----------------------+---------------------+
    |  nene.jpg |       Waterfowl       |  0.3437104821205139 |
    |  nene.jpg | Shorebirds/Waterbirds |  0.3415525555610657 |
    |  nene.jpg |         Cranes        | 0.17063717544078827 |
    +-----------+-----------------------+---------------------+
    ```
