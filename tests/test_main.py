import unittest
from unittest.mock import mock_open, patch
import argparse
import pandas as pd
from bioclip.__main__ import parse_args, Rank, create_classes_str, main, parse_bins_csv


class TestParser(unittest.TestCase):
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
        self.assertEqual(args.log, None)

        args = parse_args(['predict', 'image.jpg', 'image2.png'])
        self.assertEqual(args.command, 'predict')
        self.assertEqual(args.image_file, ['image.jpg', 'image2.png'])
        self.assertEqual(args.log, None)

        # test tree of life version of predict
        args = parse_args(['predict', 'image.jpg', '--format', 'table', '--output', 'output.csv', '--rank', 'genus', '--k', '10', '--device', 'cuda', '--log', 'log.txt'])
        self.assertEqual(args.command, 'predict')
        self.assertEqual(args.image_file, ['image.jpg'])
        self.assertEqual(args.format, 'table')
        self.assertEqual(args.output, 'output.csv')
        self.assertEqual(args.rank, Rank.GENUS)
        self.assertEqual(args.k, 10)
        self.assertEqual(args.cls, None)
        self.assertEqual(args.device, 'cuda')
        self.assertEqual(args.subset, None)
        self.assertEqual(args.batch_size, 10)
        self.assertEqual(args.log, 'log.txt')

        # test tree of life subset
        args = parse_args(['predict', 'image.jpg', '--subset', 'somefile.csv', '--log', 'log.txt'])
        self.assertEqual(args.subset, 'somefile.csv')
        self.assertEqual(args.log, 'log.txt')

        # test custom class list version of predict
        args = parse_args(['predict', 'image.jpg', '--format', 'table', '--output', 'output.csv',
                           '--cls', 'class1,class2', '--device', 'cuda', '--batch-size', '5', '--log', 'log.txt'])
        self.assertEqual(args.command, 'predict')
        self.assertEqual(args.image_file, ['image.jpg'])
        self.assertEqual(args.format, 'table')
        self.assertEqual(args.output, 'output.csv')
        self.assertEqual(args.rank, None) # default ignored for the --cls variation
        self.assertEqual(args.k, None)
        self.assertEqual(args.cls, 'class1,class2')
        self.assertEqual(args.bins, None)
        self.assertEqual(args.device, 'cuda')
        self.assertEqual(args.batch_size, 5)
        self.assertEqual(args.log, 'log.txt')

        # test binning version of predict
        args = parse_args(['predict', 'image.jpg', '--format', 'table', '--output', 'output.csv', '--bins', 'bins.csv', '--device', 'cuda', '--log', 'log.txt'])
        self.assertEqual(args.command, 'predict')
        self.assertEqual(args.image_file, ['image.jpg'])
        self.assertEqual(args.format, 'table')
        self.assertEqual(args.output, 'output.csv')
        self.assertEqual(args.rank, None) # default ignored for the --cls variation
        self.assertEqual(args.k, None)
        self.assertEqual(args.cls, None)
        self.assertEqual(args.bins, 'bins.csv')
        self.assertEqual(args.device, 'cuda')
        self.assertEqual(args.log, 'log.txt')

        # test error when using --cls with --rank
        with self.assertRaises(SystemExit):
            parse_args(['predict', 'image.jpg', '--cls', 'class1,class2', '--rank', 'genus'])

        # test error when using --cls with --bins
        with self.assertRaises(SystemExit):
            parse_args(['predict', 'image.jpg', '--cls', 'class1,class2', '--bins', 'somefile.csv'])

        # not an error when using --cls with --k
        args = parse_args(['predict', 'image.jpg', '--cls', 'class1,class2', '--k', '10'])
        self.assertEqual(args.k, 10)
        self.assertEqual(args.log, None)

        # test error when using --cls with --subset
        with self.assertRaises(SystemExit):
            parse_args(['predict', 'image.jpg', '--cls', 'class1,class2', '--subset', 'somefile.cvs'])

        # test error when using --bins with --subset
        with self.assertRaises(SystemExit):
            parse_args(['predict', 'image.jpg', '--bins', 'somefile.csv', '--subset', 'somefile.cvs'])

        # test error when using --rank with --subset
        with self.assertRaises(SystemExit):
            parse_args(['predict', 'image.jpg', '--rank', 'class', '--subset', 'somefile.cvs'])

        # example showing filename
        args = parse_args(['predict', 'image.jpg', '--cls', 'classes.txt', '--k', '10', '--log', 'log.txt'])
        self.assertEqual(args.cls, 'classes.txt')
        self.assertEqual(args.log, 'log.txt')

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

        args = parse_args(['list-tol-taxa'])
        self.assertEqual(args.command, 'list-tol-taxa')

    def test_create_classes_str(self):
        data = "class1\nclass2\nclass3"
        with patch("builtins.open", mock_open(read_data=data)) as mock_file:
            self.assertEqual(create_classes_str('path/to/file'), 'class1,class2,class3')

    @patch('bioclip.__main__.predict')
    @patch('bioclip.__main__.parse_args')
    def test_predict_no_class(self, mock_parse_args, mock_predict):
        mock_parse_args.return_value = argparse.Namespace(command='predict', image_file='image.jpg', format='csv',
                                                          output='stdout', rank=Rank.SPECIES, k=5, cls=None, device='cpu',
                                                          model=None, pretrained=None, bins=None, subset=None,
                                                          batch_size=None, log=None)
        main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str=None, rank=Rank.SPECIES,
                                        bins_path=None, k=5, device='cpu', model_str=None, pretrained_str=None,
                                        subset=None, batch_size=None, log=None)

    @patch('bioclip.__main__.predict')
    @patch('bioclip.__main__.parse_args')
    @patch('bioclip.__main__.os')
    def test_predict_class_list(self, mock_os, mock_parse_args, mock_predict):
        mock_os.path.exists.return_value = False
        mock_parse_args.return_value = argparse.Namespace(command='predict', image_file='image.jpg', format='csv',
                                                          output='stdout', rank=Rank.SPECIES, k=5, cls='dog,fish,bird',
                                                          device='cpu', model=None, pretrained=None, bins=None,
                                                          subset=None, batch_size=None, log=None)
        main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str='dog,fish,bird', rank=Rank.SPECIES,
                                        bins_path=None, k=5, device='cpu', model_str=None, pretrained_str=None,
                                        subset=None, batch_size=None, log=None)

    @patch('bioclip.__main__.predict')
    @patch('bioclip.__main__.parse_args')
    @patch('bioclip.__main__.os')
    def test_predict_class_file(self, mock_os, mock_parse_args, mock_predict):
        mock_os.path.exists.return_value = True
        mock_parse_args.return_value = argparse.Namespace(command='predict', image_file='image.jpg', format='csv', 
                                                          output='stdout', rank=Rank.SPECIES, k=5, cls='somefile.txt',
                                                          device='cpu', model=None, pretrained=None, bins=None,
                                                          subset=None, batch_size=None, log=None)
        with patch("builtins.open", mock_open(read_data='dog\nfish\nbird')) as mock_file:
            main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str='dog,fish,bird', rank=Rank.SPECIES,
                                        bins_path=None, k=5, device='cpu', model_str=None, pretrained_str=None,
                                        subset=None, batch_size=None, log=None)

    @patch('bioclip.__main__.predict')
    @patch('bioclip.__main__.parse_args')
    @patch('bioclip.__main__.os')
    def test_predict_bins(self, mock_os, mock_parse_args, mock_predict):
        mock_os.path.exists.return_value = True
        mock_parse_args.return_value = argparse.Namespace(command='predict', image_file='image.jpg', format='csv', 
                                                          output='stdout', rank=None, k=5, cls=None, 
                                                          device='cpu', model=None, pretrained=None,
                                                          bins='some.csv', subset=None,
                                                          batch_size=None, log=None)
        with patch("builtins.open", mock_open(read_data='dog\nfish\nbird')) as mock_file:
            main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str=None, rank=None,
                                        bins_path='some.csv', k=5, device='cpu', model_str=None,
                                        pretrained_str=None, subset=None, batch_size=None, log=None)

    @patch('bioclip.__main__.predict')
    @patch('bioclip.__main__.parse_args')
    @patch('bioclip.__main__.os')
    def test_predict_subset(self, mock_os, mock_parse_args, mock_predict):
        mock_os.path.exists.return_value = True
        mock_parse_args.return_value = argparse.Namespace(command='predict', image_file='image.jpg', format='csv',
                                                          output='stdout', rank=None, k=5, cls=None,
                                                          device='cpu', model=None, pretrained=None,
                                                          bins=None, subset='somefile.csv',
                                                          batch_size=None, log=None)
        main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str=None, rank=None,
                                        bins_path=None, k=5, device='cpu', model_str=None,
                                        pretrained_str=None, subset='somefile.csv', batch_size=None, log=None)

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
