import unittest
from bioclip.__main__ import parse_args, Rank


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
