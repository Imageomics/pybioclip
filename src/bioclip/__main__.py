from bioclip import TreeOfLifeClassifier, Rank, CustomLabelsClassifier
import open_clip as oc
import json
import sys
import prettytable as pt
import pandas as pd
import argparse


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


def predict(image_file: list[str],
            format: str,
            output: str,
            cls_str: str,
            rank: Rank,
            k: int,
            **kwargs):
    if cls_str:
        classifier = CustomLabelsClassifier(cls_ary=cls_str.split(','), **kwargs)
        predictions = classifier.predict(image_paths=image_file, k=k)
        write_results(predictions, format, output)
    else:
        classifier = TreeOfLifeClassifier(**kwargs)
        predictions = classifier.predict(image_paths=image_file, rank=rank, k=k)
        write_results(predictions, format, output)


def embed(image_file: list[str], output: str, **kwargs):
    classifier = TreeOfLifeClassifier(**kwargs)
    images_dict = {}
    data = {
        "model": classifier.model_str,
        "embeddings": images_dict
    }
    for image_path in image_file:
        features = classifier.create_image_features_for_path(image_path=image_path, normalize=False)
        images_dict[image_path] = features.tolist()
    if output == 'stdout':
        print(json.dumps(data, indent=4))
    else:
        with open(output, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def create_parser():
    parser = argparse.ArgumentParser(prog='bioclip', description='BioCLIP command line interface')
    subparsers = parser.add_subparsers(title='commands', dest='command')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Use BioCLIP to generate predictions for image files.')
    predict_parser.add_argument('image_file', nargs='+', help='input image file(s)')
    predict_parser.add_argument('--format', choices=['table', 'csv'], default='csv', help='format of the output, default: csv')
    predict_parser.add_argument('--output', default='stdout', help='print output to file, default: stdout')
    predict_parser.add_argument('--rank', choices=['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'],
                                help='rank of the classification, default: species (when)')
    predict_parser.add_argument('--k', type=int, help='number of top predictions to show, default: 5')
    predict_parser.add_argument('--cls', help='comma separated list of classes to predict, when specified the --rank and --k arguments are not allowed')
    predict_parser.add_argument('--device', help='device to use (cpu or cuda or mps), default: cpu', default='cpu')
    predict_parser.add_argument('--model', help='model identifier (see open_clip); default: hf-hub:imageomics/bioclip')
    predict_parser.add_argument('--pretrained', help='pretrained model checkpoint as tag or file, depends on model')

    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Use BioCLIP to generate embeddings for image files.')
    embed_parser.add_argument('image_file', nargs='+', help='input image file(s)')
    embed_parser.add_argument('--output', default='stdout', help='print output to file, default: stdout')
    embed_parser.add_argument('--device', help='device to use (cpu or cuda or mps), default: cpu', default='cpu')
    embed_parser.add_argument('--model', help='model identifier (see open_clip); default: hf-hub:imageomics/bioclip')
    embed_parser.add_argument('--pretrained', help='pretrained model checkpoint as tag or file, depends on model')

    # List command
    list_parser = subparsers.add_parser('list-models', help='List available models and pretrained model checkpoints.')
    list_parser.add_argument('--model', help='list available pretrained model checkpoint(s) for model')

    return parser


def parse_args(input_args=None):
    args = create_parser().parse_args(input_args)
    if args.command == 'predict':
        if args.cls:
            # custom class list mode
            if args.rank:
                raise ValueError("Cannot use --cls with --rank")
        else:
            # tree of life class list mode
            if args.model or args.pretrained:
                raise ValueError("Custom model or checkpoints currently not supported for Tree-of-Life prediction")
            if not args.rank:
                args.rank = 'species'
            args.rank = Rank[args.rank.upper()]
            if not args.k:
                args.k = 5
    return args


def main():
    args = parse_args()
    if args.command == 'embed':
        embed(args.image_file,
              args.output,
              device=args.device,
              model_str=args.model,
              pretrained_str=args.pretrained)
    elif args.command == 'predict':
        predict(args.image_file,
                format=args.format,
                output=args.output,
                cls_str=args.cls,
                rank=args.rank,
                k=args.k,
                device=args.device,
                model_str=args.model,
                pretrained_str=args.pretrained)
    elif args.command == 'list-models':
        if args.model:
            for tag in oc.list_pretrained_tags_by_model(args.model):
                print(tag)
        else:
            for model_str in oc.list_models():
                print(f"\t{model_str}")
    else:
        raise ValueError("Invalid command")


if __name__ == '__main__':
    main()
