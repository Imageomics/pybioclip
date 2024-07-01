from bioclip import TreeOfLifeClassifier, Rank, CustomLabelsClassifier
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

def predict(image_file: list[str], format: str,  output: str,
             cls_str: str, device: str,  rank: Rank, k: int):
    if cls_str:
        classifier = CustomLabelsClassifier(device=device)
        data = []
        for image_path in image_file:
            data.extend(classifier.predict(image_path=image_path, cls_ary=cls_str.split(',')))
        write_results(data, format, output)
    else:
        classifier = TreeOfLifeClassifier(device=device)
        data = []
        for image_path in image_file:
            data.extend(classifier.predict(image_path=image_path, rank=rank, k=k))
        write_results(data, format, output)


def embed(image_file: list[str], output: str, device: str):
    classifier = TreeOfLifeClassifier(device=device)
    images_dict = {}
    data = {
        "model": classifier.model_str,
        "embeddings": images_dict
    }
    for image_path in image_file:
        features = classifier.get_image_features(image_path)[0]
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

    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Use BioCLIP to generate embeddings for image files.')
    embed_parser.add_argument('image_file', nargs='+', help='input image file(s)')
    embed_parser.add_argument('--output', default='stdout', help='print output to file, default: stdout')
    embed_parser.add_argument('--device', help='device to use (cpu or cuda or mps), default: cpu', default='cpu')

    return parser


def parse_args(input_args=None):
    args = create_parser().parse_args(input_args)
    if args.command == 'predict':
        if args.cls:
            # custom class list mode
            if args.rank or args.k:
                raise ValueError("Cannot use --cls with --rank or --k")
        else:
            # tree of life class list mode
            if not args.rank:
                args.rank = 'species'
            args.rank = Rank[args.rank.upper()]
            if not args.k:
                args.k = 5
    return args


def main():
    args = parse_args()
    if args.command == 'embed':
        embed(args.image_file, args.output, args.device)
    elif args.command == 'predict':
        predict(args.image_file, args.format, args.output, args.cls, args.device, args.rank, args.k)
    else:
        raise ValueError("Invalid command")


if __name__ == '__main__':
    main()
