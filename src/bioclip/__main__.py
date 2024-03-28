"""Usage: bioclip predict [options] IMAGE_FILE 

Use BioCLIP to generate predictions for an IMAGE_FILE.

Arguments:
  IMAGE_FILE           input image file

Options:
  -h --help
  --format=FORMAT      format of the output (table, json, or csv) [default: table]
  --rank=RANK          rank of the classification (kingdom, phylum, class, order, family, genus, species) [default: species] 
  --k=K                number of top predictions to show [default: 5]
  --cls=CLS            comma separated list of classes to predict, when specified the --rank and --k arguments are ignored [default: all]
  --output=OUTFILE     save output to a filename instead of printing it [default: stdout]

"""
from docopt import docopt
from bioclip import predict_classification, predict_classifications_from_list, Rank
import json
import sys
import prettytable as pt
import csv


def write_results(result, format, outfile):
    if format == 'table':
        table = pt.PrettyTable()
        table.field_names = ['Taxon', 'Probability']
        for taxon, prob in result.items():
            table.add_row([taxon, prob])
        outfile.write(str(table))
        outfile.write('\n')
    elif format == 'json':
        json.dump(result, outfile, indent=2)
    elif format == 'csv':
        writer = csv.writer(outfile)
        writer.writerow(['Taxon', 'Probability'])
        for taxon, prob in result.items():
            writer.writerow([taxon, prob])
    else:
        raise ValueError(f"Invalid format: {format}")


def main():
    # execute only if run as the entry point into the program
    x = docopt(__doc__)  # parse arguments based on docstring above
    format = x['--format']
    output = x['--output']
    image_file = x['IMAGE_FILE']
    cls = x['--cls']
    if not format in ['table', 'json', 'csv']:
        raise ValueError(f"Invalid format: {format}")
    rank = Rank[x['--rank'].upper()]
    if cls == 'all':
        result = predict_classification(img=image_file,
                                        rank=rank,
                                        k=int(x['--k']))
    else:
        result = predict_classifications_from_list(img=image_file, 
                                                   cls_ary=cls.split(','))
    outfile = sys.stdout
    if output == 'stdout':
        write_results(result, format, sys.stdout)
    else:
        with open(output, 'w') as outfile:
            write_results(result, format, outfile)


if __name__ == '__main__':
    main()
