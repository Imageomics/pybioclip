import json
import csv
import sys
import prettytable as pt
from bioclip.predict import get_tol_classification_labels, PRED_CLASSICATION_KEY, PRED_SCORE_KEY


OUTPUT_FILE_NAME_COLUMN_NAME = "file_name"
OUTPUT_LABEL_COLUMN_NAME = "label"
OUTPUT_SCORE_COLUMN_NAME = "score"
LEFT_ALIGN_COLUMN = "l"
RIGHT_ALIGN_COLUMN = "r"


def get_tol_output_labels(rank):
    field_names = [OUTPUT_FILE_NAME_COLUMN_NAME]
    field_names.extend(get_tol_classification_labels(rank))
    field_names.append(OUTPUT_SCORE_COLUMN_NAME)
    return field_names


def get_tol_output_values(item: dict):
    values = [item[OUTPUT_FILE_NAME_COLUMN_NAME]]
    values.extend(item[PRED_CLASSICATION_KEY].values())
    values.append(item[PRED_SCORE_KEY])
    return values


def write_tree_of_life_results(outfile, pred_list, format, rank):
    if format == 'table':
        table = pt.PrettyTable()
        table.field_names = get_tol_output_labels(rank)
        for field in table.field_names:
            if field == OUTPUT_SCORE_COLUMN_NAME:
                table.align[field] = RIGHT_ALIGN_COLUMN
            else:
                table.align[field] = LEFT_ALIGN_COLUMN
        for prediction in pred_list:
            table.add_row(get_tol_output_values(prediction))
        outfile.write(str(table))
        outfile.write('\n')
    elif format == 'json':
        json.dump(pred_list, outfile, indent=2)
    elif format == 'csv':
        writer = csv.writer(outfile)
        writer.writerow(get_tol_output_labels(rank))
        for prediction in pred_list:
            writer.writerow(get_tol_output_values(prediction))
    else:
        raise ValueError(f"Invalid format: {format}")


def write_custom_labels_results(outfile, pred_list, format, cls_ary):
    output_column_names = [OUTPUT_LABEL_COLUMN_NAME, OUTPUT_SCORE_COLUMN_NAME]
    if format == 'table':
        table = pt.PrettyTable()
        table.field_names = output_column_names
        table.align[OUTPUT_LABEL_COLUMN_NAME] = LEFT_ALIGN_COLUMN
        table.align[OUTPUT_SCORE_COLUMN_NAME] = RIGHT_ALIGN_COLUMN
        for prediction in pred_list:
            table.add_row([prediction[PRED_CLASSICATION_KEY], prediction[PRED_SCORE_KEY]])
        outfile.write(str(table))
        outfile.write('\n')
    elif format == 'json':
        json.dump(pred_list, outfile, indent=2)
    elif format == 'csv':
        writer = csv.writer(outfile)
        writer.writerow(output_column_names)
        for prediction in pred_list:
            writer.writerow([prediction[PRED_CLASSICATION_KEY], prediction[PRED_SCORE_KEY]])
    else:
        raise ValueError(f"Invalid format: {format}")
