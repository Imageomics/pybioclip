from bioclip import TreeOfLifeClassifier, Rank, CustomLabelsClassifier, CustomLabelsBinningClassifier
from .__about__ import __version__ as pybioclip_version
from .predict import BIOCLIP_MODEL_STR, TOL_MODELS, ensure_tol_supported_model, get_rank_labels
from .recorder import attach_prediction_recorder, save_recorded_predictions, verify_recorder_path
import open_clip as oc
import os
import json
import sys
import prettytable as pt
import pandas as pd
import argparse
from typing import Union

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


def parse_bins_csv(bins_path):
    if not os.path.exists(bins_path):
        raise FileNotFoundError(f"File not found: {bins_path}")
    bin_df = pd.read_csv(bins_path, index_col=0)
    if len(bin_df.columns) == 0:
        raise ValueError("CSV file must have at least two columns.")
    return bin_df[bin_df.columns[0]].to_dict()


def predict(image_file: list[str],
            format: str,
            output: str,
            cls_str: str,
            rank: Rank,
            bins_path: str,
            k: int,
            subset: str,
            batch_size: int,
            log: Union[str, None],
            **kwargs):
    if log:
        verify_recorder_path(log)
    if cls_str:
        classifier = CustomLabelsClassifier(cls_ary=cls_str.split(','), **kwargs)
        if log:
            attach_prediction_recorder(classifier, classes=cls_str)
        predictions = classifier.predict(images=image_file, k=k, batch_size=batch_size)
        write_results(predictions, format, output)
    elif bins_path:
        cls_to_bin = parse_bins_csv(bins_path)
        classifier = CustomLabelsBinningClassifier(cls_to_bin=cls_to_bin, **kwargs)
        if log:
            attach_prediction_recorder(classifier, bins_path=bins_path)
        predictions = classifier.predict(images=image_file, k=k, batch_size=batch_size)
        write_results(predictions, format, output)
    else:
        classifier = TreeOfLifeClassifier(**kwargs)
        if log:
            attach_prediction_recorder(classifier, tree_of_life_version=classifier.get_tol_repo_id(), subset=subset)
        if subset:
            filter = classifier.create_taxa_filter_from_csv(subset)
            classifier.apply_filter(filter)
        predictions = classifier.predict(images=image_file, rank=rank, k=k, batch_size=batch_size)
        write_results(predictions, format, output)
    if log:
        save_recorded_predictions(classifier, log)


def embed(image_file: list[str], output: str, **kwargs):
    classifier = TreeOfLifeClassifier(**kwargs)
    images_dict = {}
    data = {
        "model": classifier.model_str,
        "embeddings": images_dict
    }
    for image_path in image_file:
        features = classifier.create_image_features_for_image(image=image_path, normalize=False)
        images_dict[image_path] = features.tolist()
    if output == 'stdout':
        print(json.dumps(data, indent=4))
    else:
        with open(output, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def create_parser():
    parser = argparse.ArgumentParser(prog='bioclip', description='BioCLIP command line interface')
    parser.add_argument('--version', action='version', version=f'pybioclip {pybioclip_version}')
    subparsers = parser.add_subparsers(title='commands', dest='command')

    device_arg = {'default':'cpu', 'help': 'device to use (cpu or cuda or mps), default: cpu'}
    output_arg = {'default': 'stdout', 'help': 'print output to file, default: stdout'}
    model_arg = {'help': f'model identifier (see command list-models); default: {BIOCLIP_MODEL_STR}'}
    pretrained_arg = {'help': 'pretrained model checkpoint as tag or file, depends on model; '
                              'needed only if more than one is available (see command list-models)'}
    batch_size_arg = {'default': 10, 'type': int,
                      'help': 'Number of images to process in a batch, default: 10'}

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Use BioCLIP to generate predictions for image files.')
    predict_parser.add_argument('image_file', nargs='+', help='input image file(s)')
    predict_parser.add_argument('--format', choices=['table', 'csv'], default='csv', help='format of the output, default: csv')
    predict_parser.add_argument('--output', **output_arg)
    cls_group = predict_parser.add_mutually_exclusive_group(required=False)
    cls_group.add_argument('--rank', choices=['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'],
                                help='rank of the classification, default: species, when specified the --cls, --bins, and --subset arguments are not allowed.')
    cls_help = "classes to predict: either a comma separated list or a path to a text file of classes (one per line), when specified the --rank, --bins, and --subset arguments are not allowed."
    cls_group.add_argument('--cls', help=cls_help)
    cls_group.add_argument('--bins', help='path to CSV file with two columns with the first being classes and second being bin names, when specified the --rank, --cls, and --subset arguments are not allowed.')
    subset_labels = ','.join(get_rank_labels())
    SUBSET_HELP = f"path to CSV file used to subset the tree of life embeddings. CSV first column must be named one of {subset_labels}. When specified the --rank, --bins, and --cls arguments are not allowed."
    cls_group.add_argument('--subset', help=SUBSET_HELP)
    predict_parser.add_argument('--k', type=int, help='number of top predictions to show, default: 5')

    predict_parser.add_argument('--device', **device_arg)
    predict_parser.add_argument('--model', **model_arg)
    predict_parser.add_argument('--pretrained', **pretrained_arg)
    predict_parser.add_argument('--batch-size', **batch_size_arg)
    predict_parser.add_argument(
        '--log',
        metavar='LOG_FILE',
        type=str,
        default=None,
        help=(
            "Path to a file for recording prediction logs. "
            "If the file extension is '.json', logs are written in machine-readable JSON for building a provenance chain; otherwise, logs are appended in a human-readable text format. "
            "If not specified, no log is written."
        )
    )

    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Use BioCLIP to generate embeddings for image files.')
    embed_parser.add_argument('image_file', nargs='+', help='input image file(s)')
    embed_parser.add_argument('--output', **output_arg)
    embed_parser.add_argument('--device', **device_arg)
    embed_parser.add_argument('--model', **model_arg)
    embed_parser.add_argument('--pretrained', **pretrained_arg)

    # List command
    list_parser = subparsers.add_parser('list-models',
                                        help='List available models and pretrained model checkpoints.',
                                        description=
                                             'Note that this will only list models known to open_clip; '
                                             'any model identifier loadable by open_clip, such as from hf-hub, file, etc '
                                             'should also be usable for --model in the embed and predict commands. '
                                             f'(The default model {BIOCLIP_MODEL_STR} is one example.)')
    list_parser.add_argument('--model', help='list available tags for pretrained model checkpoint(s) for specified model')

    # List TOL taxa command
    subparsers.add_parser('list-tol-taxa', help='Print a CSV of the taxa embedding labels included with the tree of life model to the terminal.')

    return parser


def parse_args(input_args=None):
    args = create_parser().parse_args(input_args)
    if args.command == 'predict':
        if not args.cls and not args.bins:
            # tree of life class list mode
            if args.pretrained:
                raise ValueError("Custom checkpoints are currently not supported for TreeOfLife prediction")
            if args.model:
                ensure_tol_supported_model(args.model)
            if not args.rank:
                args.rank = 'species'
            args.rank = Rank[args.rank.upper()]
            if not args.k:
                args.k = 5
    return args


def create_classes_str(cls_file_path):
    """Reads a file with one class per line and returns a comma separated string of classes"""
    with open(cls_file_path, 'r') as cls_file:
        cls_str = [item.strip() for item in cls_file.readlines()]
    return ",".join(cls_str)


def main():
    # Prevent UnicodeEncodeError on Windows
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    args = parse_args()
    if args.command == 'embed':
        embed(args.image_file,
              args.output,
              device=args.device,
              model_str=args.model,
              pretrained_str=args.pretrained)
    elif args.command == 'predict':
        cls_str = args.cls
        if args.cls and os.path.exists(args.cls):
            cls_str = create_classes_str(args.cls)
        predict(args.image_file,
                format=args.format,
                output=args.output,
                cls_str=cls_str,
                rank=args.rank,
                bins_path=args.bins,
                k=args.k,
                device=args.device,
                model_str=args.model,
                pretrained_str=args.pretrained,
                subset=args.subset,
                batch_size=args.batch_size,
                log=args.log)
    elif args.command == 'list-models':
        if args.model:
            for tag in oc.list_pretrained_tags_by_model(args.model):
                print(tag)
        else:
            for model_str in list(TOL_MODELS.keys()) + oc.list_models():
                print(f"\t{model_str}")
    elif args.command == 'list-tol-taxa':
        classifier = TreeOfLifeClassifier()
        df = classifier.get_label_data()
        # Removing newline from print since to_csv already adds one
        print(df.to_csv(index=False), end='')
    else:
        create_parser().print_help()


if __name__ == '__main__':
    main()
