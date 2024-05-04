"""Usage: bioclip predict [options] [IMAGE_FILE...]

Use BioCLIP to generate predictions for an IMAGE_FILE.

Arguments:
  IMAGE_FILE           input image file

Options:
  -h --help
  --format=FORMAT      format of the output (table or csv) [default: csv]
  --rank=RANK          rank of the classification (kingdom, phylum, class, order, family, genus, species) [default: species] 
  --k=K                number of top predictions to show [default: 5]
  --cls=CLS            comma separated list of classes to predict, when specified the --rank and --k arguments are ignored [default: all]
  --device=DEVICE      device to use for prediction (cpu or cuda or mps) [default: cpu]
  --output=OUTFILE     print output to file OUTFILE [default: stdout]

"""
from docopt import docopt
from bioclip import TreeOfLifeClassifier, Rank, CustomLabelsClassifier
import json
import sys
import prettytable as pt
import csv
import pandas as pd


def write_results(data, format, output):
    df = pd.DataFrame(data)
    if output == 'stdout':
        write_results_to_file(df, format, sys.stdout)
    else:
        with open(output, 'w') as outfile:
            write_results_to_file(df, format, outfile)


def write_results_to_file(df, format, outfile):
    if format == 'table':
        table = pt.PrettyTable()
        table.field_names = df.columns
        for index, row in df.iterrows():
            table.add_row(row)
        outfile.write(str(table))
        outfile.write('\n')
    elif format == 'csv':
        df.to_csv(outfile, index=False)
    else:
        raise ValueError(f"Invalid format: {format}")


def main():
    # execute only if run as the entry point into the program
    x = docopt(__doc__)  # parse arguments based on docstring above
    format = x['--format']
    output = x['--output']
    image_file = x['IMAGE_FILE']
    device = 'cpu'
    if x['--device']:
        device = x['--device']
    cls = x['--cls']
    if not format in ['table', 'csv']:
        raise ValueError(f"Invalid format: {format}")
    rank = Rank[x['--rank'].upper()]
    if cls == 'all':
        classifier = TreeOfLifeClassifier(device=device)
        data = []
        for image_path in image_file:
            data.extend(classifier.predict(image_path=image_path, rank=rank, k=int(x['--k'])))
        write_results(data, format, output)
    else:
        classifier = CustomLabelsClassifier(device=device)
        data = []
        for image_path in image_file:
            data.extend(classifier.predict(image_path=image_path, cls_ary=cls.split(',')))
        write_results(data, format, output)


if __name__ == '__main__':
    main()
