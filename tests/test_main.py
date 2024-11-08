import unittest
from unittest.mock import mock_open, patch, MagicMock
import argparse
import pandas as pd
from enum import Enum
from bioclip.predict import Rank
from bioclip.__main__ import parse_args, create_classes_str, main, parse_bins_csv


class TestParser(unittest.TestCase):

    def test_parse_args_lazy_import(self):
        """Test that Rank is only imported when needed"""
        # Should not import Rank
        with patch('bioclip.predict.Rank') as mock_rank:
            args = parse_args(['embed', 'image.jpg'])
            mock_rank.assert_not_called()
            
        # Should not import Rank when using --cls
        with patch('bioclip.predict.Rank') as mock_rank:
            args = parse_args(['predict', 'image.jpg', '--cls', 'class1,class2'])
            mock_rank.assert_not_called()

        # Should not import Rank when using --bins
        with patch('bioclip.predict.Rank') as mock_rank:
            args = parse_args(['predict', 'image.jpg', '--bins', 'bins.csv'])
            mock_rank.assert_not_called()

        # Should import Rank for tree-of-life prediction
        with patch('bioclip.predict.Rank', Rank) as mock_rank:
            args = parse_args(['predict', 'image.jpg'])
            self.assertEqual(args.rank, Rank.SPECIES)

def test_list_models_lazy_import(self):
    """Test that open_clip is only imported when the list-models command is used"""
    # Should import open_clip for list-models
    with patch('bioclip.__main__.open_clip', create=True) as mock_oc:
        mock_parse_args = MagicMock(return_value=argparse.Namespace(
            command='list-models',
            model=None
        ))
        with patch('bioclip.__main__.parse_args', mock_parse_args), \
             patch('builtins.print'):  # prevent actual printing
            main()
            mock_oc.list_models.assert_called_once()

    # Should call list_pretrained_tags when model specified 
    with patch('bioclip.__main__.open_clip', create=True) as mock_oc:
        mock_parse_args = MagicMock(return_value=argparse.Namespace(
            command='list-models',
            model='somemodel'
        ))
        with patch('bioclip.__main__.parse_args', mock_parse_args), \
             patch('builtins.print'):  # prevent actual printing
            main()
            mock_oc.list_pretrained_tags_by_model.assert_called_once_with('somemodel')

    def test_predict_lazy_imports(self):
        """Test that classifier classes are only imported when needed"""
        # For cls_str path
        with patch('bioclip.predict.TreeOfLifeClassifier') as mock_tree, \
            patch('bioclip.predict.CustomLabelsClassifier') as mock_custom, \
            patch('bioclip.predict.CustomLabelsBinningClassifier') as mock_binning:
            mock_parse_args = MagicMock(return_value=argparse.Namespace(
                command='predict',
                image_file=['image.jpg'],
                format='csv',
                output='stdout',
                cls='cat,dog',
                bins=None,
                device='cpu',
                model=None,
                pretrained=None,
                k=5,
                rank=None
            ))
            with patch('bioclip.__main__.parse_args', mock_parse_args):
                with patch('bioclip.__main__.write_results'):  # Prevent actual write
                    main()
                    mock_custom.assert_called()
                    mock_tree.assert_not_called()
                    mock_binning.assert_not_called()

        # For bins path
        with patch('bioclip.predict.TreeOfLifeClassifier') as mock_tree, \
            patch('bioclip.predict.CustomLabelsClassifier') as mock_custom, \
            patch('bioclip.predict.CustomLabelsBinningClassifier') as mock_binning:
            mock_parse_args = MagicMock(return_value=argparse.Namespace(
                command='predict',
                image_file=['image.jpg'],
                format='csv',
                output='stdout',
                cls=None,
                bins='bins.csv',
                device='cpu',
                model=None,
                pretrained=None,
                k=5,
                rank=None
            ))
            with patch('bioclip.__main__.parse_args', mock_parse_args), \
                patch('bioclip.__main__.parse_bins_csv', return_value={}), \
                patch('bioclip.__main__.write_results'):
                main()
                mock_binning.assert_called()
                mock_tree.assert_not_called()
                mock_custom.assert_not_called()

        # For default (TreeOfLifeClassifier) path
        with patch('bioclip.predict.TreeOfLifeClassifier') as mock_tree, \
            patch('bioclip.predict.CustomLabelsClassifier') as mock_custom, \
            patch('bioclip.predict.CustomLabelsBinningClassifier') as mock_binning:
            mock_parse_args = MagicMock(return_value=argparse.Namespace(
                command='predict',
                image_file=['image.jpg'],
                format='csv',
                output='stdout',
                cls=None,
                bins=None,
                device='cpu',
                model=None,
                pretrained=None,
                k=5,
                rank=Rank.SPECIES
            ))
            with patch('bioclip.__main__.parse_args', mock_parse_args), \
                patch('bioclip.__main__.write_results'):
                main()
                mock_tree.assert_called()
                mock_custom.assert_not_called()
                mock_binning.assert_not_called()

    def test_embed_lazy_imports(self):
        """Test that TreeOfLifeClassifier is only imported for embed command"""
        class MockTensor:
            def tolist(self):
                return [1.0, 2.0, 3.0]
        
        with patch('bioclip.predict.TreeOfLifeClassifier') as mock_clf:
            # Mock the classifier instance
            mock_clf_instance = MagicMock()
            mock_clf.return_value = mock_clf_instance
            
            # Make create_image_features_for_image return our mock tensor
            mock_clf_instance.create_image_features_for_image.return_value = MockTensor()
            mock_clf_instance.model_str = "test-model"
            
            mock_parse_args = MagicMock(return_value=argparse.Namespace(
                command='embed',
                image_file=['image.jpg'],
                output='stdout',
                device='cpu',
                model=None,
                pretrained=None
            ))
            with patch('bioclip.__main__.parse_args', mock_parse_args), \
                patch('builtins.print'):  # prevent actual printing to stdout
                main()
                mock_clf.assert_called_once()
                mock_clf_instance.create_image_features_for_image.assert_called_once()

    def test_parse_args(self):

        args = parse_args(['predict', 'image.jpg'])
        self.assertEqual(args.command, 'predict')
        self.assertEqual(args.image_file, ['image.jpg'])
        self.assertEqual(args.format, 'csv')
        self.assertEqual(args.output, 'stdout')
        self.assertEqual(args.rank, Rank.SPECIES)
        self.assertEqual(args.k, 5)
        self.assertEqual(args.cls, None)
        self.assertEqual(args.bins, None)
        self.assertEqual(args.device, 'cpu')

        args = parse_args(['predict', 'image.jpg', 'image2.png'])
        self.assertEqual(args.command, 'predict')
        self.assertEqual(args.image_file, ['image.jpg', 'image2.png'])

        # test tree of life version of predict
        args = parse_args(['predict', 'image.jpg', '--format', 'table', '--output', 'output.csv', '--rank', 'genus', '--k', '10', '--device', 'cuda'])
        self.assertEqual(args.command, 'predict')
        self.assertEqual(args.image_file, ['image.jpg'])
        self.assertEqual(args.format, 'table')
        self.assertEqual(args.output, 'output.csv')
        self.assertEqual(args.rank, Rank.GENUS)
        self.assertEqual(args.k, 10)
        self.assertEqual(args.cls, None)
        self.assertEqual(args.device, 'cuda')

        # test custom class list version of predict
        args = parse_args(['predict', 'image.jpg', '--format', 'table', '--output', 'output.csv', '--cls', 'class1,class2', '--device', 'cuda'])
        self.assertEqual(args.command, 'predict')
        self.assertEqual(args.image_file, ['image.jpg'])
        self.assertEqual(args.format, 'table')
        self.assertEqual(args.output, 'output.csv')
        self.assertEqual(args.rank, None) # default ignored for the --cls variation
        self.assertEqual(args.k, None)
        self.assertEqual(args.cls, 'class1,class2')
        self.assertEqual(args.bins, None)
        self.assertEqual(args.device, 'cuda')

        # test binning version of predict
        args = parse_args(['predict', 'image.jpg', '--format', 'table', '--output', 'output.csv', '--bins', 'bins.csv', '--device', 'cuda'])
        self.assertEqual(args.command, 'predict')
        self.assertEqual(args.image_file, ['image.jpg'])
        self.assertEqual(args.format, 'table')
        self.assertEqual(args.output, 'output.csv')
        self.assertEqual(args.rank, None) # default ignored for the --cls variation
        self.assertEqual(args.k, None)
        self.assertEqual(args.cls, None)
        self.assertEqual(args.bins, 'bins.csv')
        self.assertEqual(args.device, 'cuda')

        # test error when using --cls with --rank
        with self.assertRaises(SystemExit):
            parse_args(['predict', 'image.jpg', '--cls', 'class1,class2', '--rank', 'genus'])

        # test error when using --cls with --bins
        with self.assertRaises(SystemExit):
            parse_args(['predict', 'image.jpg', '--cls', 'class1,class2', '--bins', 'somefile.csv'])

        # not an error when using --cls with --k
        args = parse_args(['predict', 'image.jpg', '--cls', 'class1,class2', '--k', '10'])
        self.assertEqual(args.k, 10)

        # example showing filename
        args = parse_args(['predict', 'image.jpg', '--cls', 'classes.txt', '--k', '10'])
        self.assertEqual(args.cls, 'classes.txt')

        args = parse_args(['embed', 'image.jpg'])
        self.assertEqual(args.command, 'embed')
        self.assertEqual(args.image_file, ['image.jpg'])
        self.assertEqual(args.output, 'stdout')
        self.assertEqual(args.device, 'cpu')

        args = parse_args(['embed', '--output', 'data.json', '--device', 'cuda', 'image.jpg', 'image2.png'])
        self.assertEqual(args.command, 'embed')
        self.assertEqual(args.image_file, ['image.jpg', 'image2.png'])
        self.assertEqual(args.output, 'data.json')
        self.assertEqual(args.device, 'cuda')

    def test_create_classes_str(self):
        data = "class1\nclass2\nclass3"
        with patch("builtins.open", mock_open(read_data=data)) as mock_file:
            self.assertEqual(create_classes_str('path/to/file'), 'class1,class2,class3')

    @patch('bioclip.__main__.predict')
    @patch('bioclip.__main__.parse_args')
    def test_predict_no_class(self, mock_parse_args, mock_predict):
        mock_parse_args.return_value = argparse.Namespace(command='predict', image_file='image.jpg', format='csv',
                                                          output='stdout', rank=Rank.SPECIES, k=5, cls=None, device='cpu',
                                                          model=None, pretrained=None, bins=None)
        main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str=None, rank=Rank.SPECIES,
                                        bins_path=None, k=5, device='cpu', model_str=None, pretrained_str=None)

    @patch('bioclip.__main__.predict')
    @patch('bioclip.__main__.parse_args')
    @patch('bioclip.__main__.os')
    def test_predict_class_list(self, mock_os, mock_parse_args, mock_predict):
        mock_os.path.exists.return_value = False
        mock_parse_args.return_value = argparse.Namespace(command='predict', image_file='image.jpg', format='csv',
                                                          output='stdout', rank=Rank.SPECIES, k=5, cls='dog,fish,bird',
                                                          device='cpu', model=None, pretrained=None, bins=None)
        main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str='dog,fish,bird', rank=Rank.SPECIES,
                                        bins_path=None, k=5, device='cpu', model_str=None, pretrained_str=None)

    @patch('bioclip.__main__.predict')
    @patch('bioclip.__main__.parse_args')
    @patch('bioclip.__main__.os')
    def test_predict_class_file(self, mock_os, mock_parse_args, mock_predict):
        mock_os.path.exists.return_value = True
        mock_parse_args.return_value = argparse.Namespace(command='predict', image_file='image.jpg', format='csv', 
                                                          output='stdout', rank=Rank.SPECIES, k=5, cls='somefile.txt',
                                                          device='cpu', model=None, pretrained=None, bins=None)
        with patch("builtins.open", mock_open(read_data='dog\nfish\nbird')) as mock_file:
            main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str='dog,fish,bird', rank=Rank.SPECIES,
                                        bins_path=None, k=5, device='cpu', model_str=None, pretrained_str=None)

    @patch('bioclip.__main__.predict')
    @patch('bioclip.__main__.parse_args')
    @patch('bioclip.__main__.os')
    def test_predict_bins(self, mock_os, mock_parse_args, mock_predict):
        mock_os.path.exists.return_value = True
        mock_parse_args.return_value = argparse.Namespace(command='predict', image_file='image.jpg', format='csv', 
                                                          output='stdout', rank=None, k=5, cls=None, 
                                                          device='cpu', model=None, pretrained=None,
                                                          bins='some.csv')
        with patch("builtins.open", mock_open(read_data='dog\nfish\nbird')) as mock_file:
            main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str=None, rank=None,
                                        bins_path='some.csv', k=5, device='cpu', model_str=None, pretrained_str=None)
    @patch('bioclip.__main__.os.path')
    def test_parse_bins_csv_file_missing(self, mock_path):
        mock_path.exists.return_value = False
        with self.assertRaises(FileNotFoundError) as raised_exception:
            parse_bins_csv("somefile.csv")
        self.assertEqual(str(raised_exception.exception), 'File not found: somefile.csv')

    @patch('bioclip.__main__.pd')
    @patch('bioclip.__main__.os.path')
    def test_parse_bins_csv(self, mock_path, mock_pd):
        mock_path.exists.return_value = True
        data = {'bin': ['a', 'b']}
        mock_pd.read_csv.return_value = pd.DataFrame(data=data, index=['dog', 'cat'])
        with patch("builtins.open", mock_open(read_data='dog\nfish\nbird')) as mock_file:
            cls_to_bin = parse_bins_csv("somefile.csv")
        self.assertEqual(cls_to_bin, {'cat': 'b', 'dog': 'a'})
