import builtins
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from bioclip.commands import embed

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from bioclip._parquet import write_embeddings_parquet
except ModuleNotFoundError as error:
    if error.name != 'pyarrow':
        raise
    pa = None


@unittest.skipIf(pa is None, 'Install pybioclip[parquet] to run Parquet export tests')
class TestParquetExport(unittest.TestCase):
    @patch('bioclip.commands.BaseClassifier')
    def test_batched_export_round_trip(self, classifier_class):
        classifier = classifier_class.return_value
        classifier.model_str = 'ViT-B-32'
        classifier.pretrained_str = 'custom-checkpoint'
        classifier.ensure_rgb_image.side_effect = lambda path: path
        classifier.create_image_features.side_effect = [
            torch.tensor([[0.1, 0.2], [3.0, 4.0]], dtype=torch.float64),
            torch.tensor([[5.0, 6.0]], dtype=torch.float64),
        ]
        images = ['leaf.jpg', 'whole-tree.jpg', 'another-leaf.jpg']
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'embeddings.parquet'
            embed(images, str(output), batch_size=2, format='parquet', model_str='ViT-B-32')
            table = pq.read_table(output)
        classifier_class.assert_called_once_with(model_str='ViT-B-32')
        self.assertEqual(table.column_names, ['uuid', 'emb'])
        self.assertEqual(table.schema.field('uuid').type, pa.string())
        self.assertEqual(table.schema.field('emb').type, pa.list_(pa.float32()))
        self.assertEqual(table['uuid'].to_pylist(), images)
        torch.testing.assert_close(
            torch.tensor(table['emb'].to_pylist()),
            torch.tensor([[0.1, 0.2], [3.0, 4.0], [5.0, 6.0]]),
        )
        self.assertEqual(json.loads(table.schema.metadata[b'pybioclip']), {
            'model': 'ViT-B-32', 'pretrained': 'custom-checkpoint', 'normalized': False,
        })
        self.assertEqual(classifier.create_image_features.call_count, 2)
        self.assertEqual(classifier.create_image_features.call_args_list[0].args[0], images[:2])
        self.assertEqual(classifier.create_image_features.call_args_list[1].args[0], images[2:])
        for call in classifier.create_image_features.call_args_list:
            self.assertIs(call.kwargs['normalize'], False)

    @patch('bioclip.commands.BaseClassifier')
    def test_duplicate_paths_match_json_dictionary_semantics(self, classifier_class):
        classifier = classifier_class.return_value
        classifier.model_str = 'model'
        classifier.pretrained_str = None
        classifier.create_image_features.side_effect = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0]]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'embeddings.parquet'
            embed(['leaf.jpg', 'tree.jpg', 'leaf.jpg'], str(output), batch_size=2,
                  format='parquet')
            table = pq.read_table(output)
        self.assertEqual(table['uuid'].to_pylist(), ['leaf.jpg', 'tree.jpg'])
        self.assertEqual(table['emb'].to_pylist(), [[5.0, 6.0], [3.0, 4.0]])

    @patch('bioclip.commands.BaseClassifier')
    def test_json_default_preserves_output(self, classifier_class):
        classifier = classifier_class.return_value
        classifier.model_str = 'model'
        classifier.create_image_features.return_value = torch.tensor([[3.0, 4.0]])
        with patch('sys.stdout', new_callable=io.StringIO) as stdout:
            embed(['leaf.jpg'], 'stdout')
        self.assertEqual(json.loads(stdout.getvalue()), {
            'model': 'model', 'embeddings': {'leaf.jpg': [3.0, 4.0]},
        })

    @patch('bioclip.commands.BaseClassifier')
    def test_parquet_stdout_fails_before_model_loading(self, classifier_class):
        with self.assertRaisesRegex(ValueError, 'requires --output'):
            embed(['leaf.jpg'], 'stdout', format='parquet')
        classifier_class.assert_not_called()

    @patch('bioclip.commands.BaseClassifier')
    def test_missing_dependency_fails_before_model_loading(self, classifier_class):
        original_import = builtins.__import__

        def missing_pyarrow(name, *args, **kwargs):
            if name == '_parquet':
                raise ModuleNotFoundError('No module named pyarrow', name='pyarrow')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=missing_pyarrow):
            with self.assertRaisesRegex(ImportError, r'pybioclip\[parquet\]'):
                embed(['leaf.jpg'], 'embeddings.parquet', format='parquet')
        classifier_class.assert_not_called()

    @patch('bioclip.commands.BaseClassifier')
    def test_unrelated_import_failure_is_not_reported_as_missing_pyarrow(self, classifier_class):
        original_import = builtins.__import__

        def broken_import(name, *args, **kwargs):
            if name == '_parquet':
                raise ModuleNotFoundError('Unexpected missing module', name='unrelated_dependency')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=broken_import):
            with self.assertRaises(ModuleNotFoundError) as error:
                embed(['leaf.jpg'], 'embeddings.parquet', format='parquet')
        self.assertEqual(error.exception.name, 'unrelated_dependency')
        classifier_class.assert_not_called()

    @patch('bioclip.commands.BaseClassifier')
    def test_inference_failure_preserves_existing_file(self, classifier_class):
        classifier_class.return_value.create_image_features.side_effect = RuntimeError('encode failed')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'embeddings.parquet'
            output.write_bytes(b'existing data')
            with self.assertRaisesRegex(RuntimeError, 'encode failed'):
                embed(['leaf.jpg'], str(output), format='parquet')
            self.assertEqual(output.read_bytes(), b'existing data')

    def test_failed_write_preserves_existing_file_and_cleans_temporary_files(self):
        def partial_write(table, output):
            Path(output).write_bytes(b'incomplete parquet')
            raise OSError('write failed')

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'embeddings.parquet'
            output.write_bytes(b'existing data')
            with patch('bioclip._parquet.pq.write_table', side_effect=partial_write):
                with self.assertRaisesRegex(OSError, 'write failed'):
                    write_embeddings_parquet({'leaf.jpg': [1.0, 2.0]}, str(output), 'model', None)
            self.assertEqual(output.read_bytes(), b'existing data')
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_mismatched_dimensions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'embeddings.parquet'
            with self.assertRaisesRegex(ValueError, 'same dimensionality'):
                write_embeddings_parquet({'a': [1.0], 'b': [2.0, 3.0]}, str(output), 'model', None)
            self.assertFalse(output.exists())

    def test_empty_export_has_explicit_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'empty.parquet'
            write_embeddings_parquet({}, str(output), 'model', None)
            table = pq.read_table(output)
        self.assertEqual(table.num_rows, 0)
        self.assertEqual(table.schema.field('uuid').type, pa.string())
        self.assertEqual(table.schema.field('emb').type, pa.list_(pa.float32()))
