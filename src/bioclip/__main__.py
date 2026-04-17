from .__about__ import __version__ as pybioclip_version
from ._constants import BIOCLIP_MODEL_STR, DEFAULT_BATCH_SIZE, get_rank_labels
import argparse
import sys


def create_parser():
    parser = argparse.ArgumentParser(prog='bioclip', description='BioCLIP command line interface')
    parser.add_argument('--version', action='version', version=f'pybioclip {pybioclip_version}')
    subparsers = parser.add_subparsers(title='commands', dest='command')

    device_arg = {'default':'cpu', 'help': 'device to use (cpu or cuda or mps), default: cpu'}
    output_arg = {'default': 'stdout', 'help': 'print output to file, default: stdout'}
    model_arg = {'help': f'model identifier (see command list-models); default: {BIOCLIP_MODEL_STR}'}
    pretrained_arg = {'help': 'pretrained model checkpoint as tag or file, depends on model; '
                              'needed only if more than one is available (see command list-models)'}
    batch_size_arg = {'default': DEFAULT_BATCH_SIZE, 'type': int,
                      'help': f'Number of images to process in a batch, default: {DEFAULT_BATCH_SIZE}'}

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
    SUBSET_HELP = f"path to CSV file used to subset the TreeOfLife taxa embeddings. CSV first column must be named one of {subset_labels}. When specified the --rank, --bins, and --cls arguments are not allowed."
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
    embed_parser.add_argument('--batch-size', **batch_size_arg)

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
    list_tol_taxa_parser = subparsers.add_parser('list-tol-taxa', help=f'Print a CSV of the taxa embedding labels included with the specified model to the terminal; default: taxa in {BIOCLIP_MODEL_STR}')
    list_tol_taxa_parser.add_argument('--model', **model_arg)

    return parser


def parse_args(input_args=None):
    args = create_parser().parse_args(input_args)
    if args.command == 'predict':
        if not args.cls and not args.bins:
            # tree of life class list mode
            from bioclip.predict import Rank, ensure_tol_supported_model
            if args.pretrained:
                raise ValueError("Custom checkpoints are currently not supported for TreeOfLife prediction")
            if args.model:
                ensure_tol_supported_model(args.model)
            if not args.rank:
                args.rank = 'species'
            args.rank = Rank[args.rank.upper()]
            if not args.k:
                args.k = 5
    elif args.command == 'list-tol-taxa':
        if args.model:
            from bioclip.predict import ensure_tol_supported_model
            ensure_tol_supported_model(args.model)
    return args


def main():
    # Prevent UnicodeEncodeError on Windows
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    args = parse_args()
    from bioclip import commands
    commands.run(args)


if __name__ == '__main__':
    main()
