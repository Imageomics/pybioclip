from bioclip import TreeOfLifeClassifier, CustomLabelsClassifier, CustomLabelsBinningClassifier
from ._constants import TOL_MODELS, Rank, DEFAULT_BATCH_SIZE
from .recorder import attach_prediction_recorder, save_recorded_predictions, verify_recorder_path
import open_clip as oc
import os
import json
import sys
import prettytable as pt
import pandas as pd
from typing import Union
from tqdm import tqdm

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


def embed(image_file: list[str], output: str, batch_size: int = DEFAULT_BATCH_SIZE, **kwargs):
    classifier = TreeOfLifeClassifier(**kwargs)
    images_dict = {}
    data = {
        "model": classifier.model_str,
        "embeddings": images_dict
    }

    total_images = len(image_file)
    with tqdm(total=total_images, unit="images") as progress_bar:
        for i in range(0, len(image_file), batch_size):
            batch_paths = image_file[i:i + batch_size]
            batch_images = [classifier.ensure_rgb_image(path) for path in batch_paths]
            batch_features = classifier.create_image_features(batch_images, normalize=False)

            for j, image_path in enumerate(batch_paths):
                images_dict[image_path] = batch_features[j].tolist()

            progress_bar.update(len(batch_paths))

    if output == 'stdout':
        print(json.dumps(data, indent=4))
    else:
        with open(output, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def create_classes_str(cls_file_path):
    """Reads a file with one class per line and returns a comma separated string of classes"""
    with open(cls_file_path, 'r', encoding="utf-8") as cls_file:
        cls_str = [item.strip() for item in cls_file.readlines()]
    return ",".join(cls_str)


def run(args):
    if args.command == 'embed':
        embed(args.image_file,
              args.output,
              batch_size=args.batch_size,
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
        classifier = TreeOfLifeClassifier(model_str=args.model)
        df = classifier.get_label_data()
        # Removing newline from print since to_csv already adds one
        print(df.to_csv(index=False), end='')
    else:
        from bioclip.__main__ import create_parser
        create_parser().print_help()
