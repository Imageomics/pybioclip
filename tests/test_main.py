import unittest
from unittest.mock import mock_open, patch
import argparse
from bioclip.__main__ import parse_args, Rank, create_classes_str, main


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
        self.assertEqual(args.device, 'cuda')

        # test error when using --cls with --rank
        with self.assertRaises(ValueError):
            parse_args(['predict', 'image.jpg', '--cls', 'class1,class2', '--rank', 'genus'])

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
                                                          model=None, pretrained=None)
        main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str=None, rank=Rank.SPECIES, k=5,
                                        device='cpu', model_str=None, pretrained_str=None)

    @patch('bioclip.__main__.predict')
    @patch('bioclip.__main__.parse_args')
    @patch('bioclip.__main__.os')
    def test_predict_class_list(self, mock_os, mock_parse_args, mock_predict):
        mock_os.path.exists.return_value = False
        mock_parse_args.return_value = argparse.Namespace(command='predict', image_file='image.jpg', format='csv',
                                                          output='stdout', rank=Rank.SPECIES, k=5, cls='dog,fish,bird',
                                                          device='cpu', model=None, pretrained=None)
        main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str='dog,fish,bird', rank=Rank.SPECIES,
                                        k=5, device='cpu', model_str=None, pretrained_str=None)

    @patch('bioclip.__main__.predict')
    @patch('bioclip.__main__.parse_args')
    @patch('bioclip.__main__.os')
    def test_predict_class_file(self, mock_os, mock_parse_args, mock_predict):
        mock_os.path.exists.return_value = True
        mock_parse_args.return_value = argparse.Namespace(command='predict', image_file='image.jpg', format='csv', 
                                                          output='stdout', rank=Rank.SPECIES, k=5, cls='somefile.txt',
                                                          device='cpu', model=None, pretrained=None)
        with patch("builtins.open", mock_open(read_data='dog\nfish\nbird')) as mock_file:
            main()
        mock_predict.assert_called_with('image.jpg', format='csv', output='stdout', cls_str='dog,fish,bird', rank=Rank.SPECIES,
                                        k=5, device='cpu', model_str=None, pretrained_str=None)
