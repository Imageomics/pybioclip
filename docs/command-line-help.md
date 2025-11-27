# Command Line Help

## bioclip predict
Use BioCLIP to generate predictions for image files.
```
usage: bioclip predict [-h] [--format {table,csv}] [--output OUTPUT]
                       [--rank {kingdom,phylum,class,order,family,genus,species} |
                        --cls CLS | --bins BINS | --subset SUBSET] [--k K]
                       [--device DEVICE] [--model MODEL] [--pretrained PRETRAINED]
                       [--batch-size BATCH_SIZE]
                       image_file [image_file ...]

positional arguments:
  image_file            input image file(s)

options:
  -h, --help            show this help message and exit
  --format {table,csv}  format of the output, default: csv
  --output OUTPUT       print output to file, default: stdout
  --rank {kingdom,phylum,class,order,family,genus,species}
                        rank of the classification, default: species, when
                        specified the --cls, --bins, and --subset arguments
                        are not allowed.
  --cls CLS             classes to predict: either a comma separated list or a
                        path to a text file of classes (one per line), when
                        specified the --rank, --bins, and --subset arguments
                        are not allowed.
  --bins BINS           path to CSV file with two columns with the first being
                        classes and second being bin names, when specified the
                        --rank, --cls, and --subset arguments are not allowed.
  --subset SUBSET       path to CSV file used to subset the TreeOfLife
                        embeddings. CSV first column must be named one of
                        kingdom,phylum,class,order,family,genus,species. When
                        specified the --rank, --bins, and --cls arguments are
                        not allowed.
  --k K                 number of top predictions to show, default: 5
  --device DEVICE       device to use (cpu or cuda or mps), default: cpu
  --model MODEL         model identifier (see command list-models);
                        default: hf-hub:imageomics/bioclip-2
  --pretrained PRETRAINED
                        pretrained model checkpoint as tag or file, depends on
                        model; needed only if more than one is available
                        (see command list-models)
  --batch-size BATCH_SIZE
                        Number of images to process in a batch, default: 10
  --log LOG_FILE        Path to a file for recording prediction logs.
                        If the file extension is '.json', the log is written
                        in JSON for building a provenance chain; otherwise, 
                        logs are appended in a human-readable text format.
                        If not specified, no log is written.
```

## bioclip embed
Use BioCLIP to generate embeddings for image files.
```
usage: bioclip embed [-h] [--output OUTPUT] [--device DEVICE] [--model MODEL]
                     [--pretrained PRETRAINED] image_file [image_file ...]

positional arguments:
  image_file            input image file(s)

options:
  -h, --help            show this help message and exit
  --output OUTPUT       print output to file, default: stdout
  --device DEVICE       device to use (cpu or cuda or mps), default: cpu
  --model MODEL         model identifier (see command list-models);
                        default: hf-hub:imageomics/bioclip-2
  --pretrained PRETRAINED
                        pretrained model checkpoint as tag or file, depends
                        on model; needed only if more than one is available
                        (see command list-models)
```

## bioclip list-models
List available models and pretrained model checkpoints.
```
usage: bioclip list-models [-h] [--model MODEL]

Note that this will only list models known to open_clip; any model identifier
loadable by open_clip, such as from hf-hub, file, etc should also be usable for
--model in the embed and predict commands.
(The default model hf-hub:imageomics/bioclip-2 is one example.)

options:
  -h, --help     show this help message and exit
  --model MODEL  list available tags for pretrained model checkpoint(s) for
                 specified model
```

## bioclip list-tol-taxa
Outputs a CSV of the taxa embedding labels included with the selected (or default) TreeOfLife model. Other models are not supported (because precomputed taxon label embeddings are not available for them). 

Note that this is a very large table and should be redirected to a file. One major use of this table is to construct and/or validate a table for the `--subset` option of the `predict` command. Because the TreeOfLife training datasets ([TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) for the original BioCLIP model, [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) for BioCLIP 2) are very different between the models, their taxon embedding labels are also different (even if the intersection of taxa is large).
```
usage: bioclip list-tol-taxa [-h] [--model MODEL]

options:
  -h, --help     show this help message and exit
  --model MODEL  model identifier (see command list-models); default: hf-hub:imageomics/bioclip-2
```
