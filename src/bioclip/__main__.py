"""Usage: bioclip predict [options] IMAGE_FILE... 

Process FILE and optionally apply correction to either left-hand side or
right-hand side.

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
import sys
from bioclip.predict import CustomLabelsClassifier, TreeOfLifeClassifier, Rank
from bioclip.output import write_tree_of_life_results, write_custom_labels_results


def tree_of_life_prediction(image_files, rank, k, output, format):
    classifier = TreeOfLifeClassifier()
    pred_list = []
    for image_file in image_files:
        single_image_pred_list = classifier.predict(image_path=image_file, rank=rank, k=k)
        for pred in single_image_pred_list:
            pred["file_name"] = image_file
            pred_list.append(pred)
    if output == 'stdout':
        write_tree_of_life_results(outfile=sys.stdout, pred_list=pred_list,
                                   format=format, rank=rank)
    else:
        with open(output, 'w') as outfile:
            write_tree_of_life_results(outfile=outfile, pred_list=pred_list,
                                       format=format, rank=rank)


def custom_labels_prediction(image_files, cls, output, format):
    classifier = CustomLabelsClassifier()
    cls_ary = cls.split(',')
    pred_list = []
    for image_file in image_files:
        single_image_pred_list = classifier.predict(image_path=image_file, cls_ary=cls_ary)
        for prediction in single_image_pred_list:
            prediction["file_name"] = image_file
            pred_list.append(prediction)
    if output == 'stdout':
        write_custom_labels_results(outfile=sys.stdout, pred_list=pred_list, 
                                    format=format, cls_ary=cls_ary)
    else:
        with open(output, 'w') as outfile:
            write_custom_labels_results(outfile=outfile, pred_list=pred_list,
                                        format=format, cls_ary=cls_ary)


def main():
    # execute only if run as the entry point into the program
    x = docopt(__doc__)  # parse arguments based on docstring above
    format = x['--format']
    output = x['--output']
    k = int(x['--k'])
    image_files = x['IMAGE_FILE']
    cls = x['--cls']
    if not format in ['table', 'json', 'csv']:
        raise ValueError(f"Invalid format: {format}")
    rank = Rank[x['--rank'].upper()]
    device = 'cpu'
    if cls == 'all':
        tree_of_life_prediction(image_files, rank, k, output, format)
    else:
        custom_labels_prediction(image_files, cls, output, format)


if __name__ == '__main__':
    main()
