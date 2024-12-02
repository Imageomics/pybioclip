# Command Line Help

## bioclip predict
```
usage: bioclip predict [-h] [--format {table,csv}] [--output OUTPUT] 
                       [--rank {kingdom,phylum,class,order,family,genus,species} |
                       --cls CLS | --bins BINS] [--k K] [--device DEVICE]
                       [--model MODEL] [--pretrained PRETRAINED]
                       image_file [image_file ...]

positional arguments:
  image_file            input image file(s)

options:
  -h, --help            show this help message and exit
  --format {table,csv}  format of the output, default: csv
  --output OUTPUT       print output to file, default: stdout
  --rank {kingdom,phylum,class,order,family,genus,species}
                        rank of the classification, default: species (when)
  --cls CLS             classes to predict: either a comma separated list or a
                        path to a text file of classes (one per line), when
                        specified the --rank and --bins arguments are not allowed.
  --bins BINS           path to CSV file with two columns with the first being
                        classes and second being bin names, when specified the
                        --cls argument is not allowed.
  --k K                 number of top predictions to show, default: 5
  --device DEVICE       device to use (cpu or cuda or mps), default: cpu
  --model MODEL         model identifier (see command list-models);
                        default: hf-hub:imageomics/bioclip
  --pretrained PRETRAINED
                        pretrained model checkpoint as tag or file, depends on
                        model; needed only if more than one is available
                        (see command list-models)
```


## bioclip embed
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
                        default: hf-hub:imageomics/bioclip
  --pretrained PRETRAINED
                        pretrained model checkpoint as tag or file, depends
                        on model; needed only if more than one is available
                        (see command list-models)
```

## bioclip list-models
```
usage: bioclip list-models [-h] [--model MODEL]

Note that this will only list models known to open_clip; any model identifier
loadable by open_clip, such as from hf-hub, file, etc should also be usable for
--model in the embed and predict commands.
(The default model hf-hub:imageomics/bioclip is one example.)

options:
  -h, --help     show this help message and exit
  --model MODEL  list available tags for pretrained model checkpoint(s) for
                 specified model
```