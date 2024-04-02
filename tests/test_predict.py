import unittest
from bioclip.predict import TreeOfLifeClassifier, Rank
from bioclip.predict import CustomLabelsClassifier
import os

DIRNAME = os.path.dirname(os.path.realpath(__file__))
EXAMPLE_CAT_IMAGE = os.path.join(DIRNAME, "images", "mycat.jpg")

class TestPredict(unittest.TestCase):
    def test_tree_of_life_classifier_species(self):
        classifier = TreeOfLifeClassifier()
        prediction_ary = classifier.predict(image_path=EXAMPLE_CAT_IMAGE, rank=Rank.SPECIES)
        self.assertEqual(len(prediction_ary), 5)
        prediction_dict = {
            'file_name': EXAMPLE_CAT_IMAGE,
            'kingdom': 'Animalia',
            'phylum': 'Chordata',
            'class': 'Mammalia',
            'order': 'Carnivora',
            'family': 'Felidae',
            'genus': 'Felis',
            'species_epithet': 'catus',
            'species': 'Felis catus',
            'common_name': 'Domestic Cat',
            'score': unittest.mock.ANY
        }
        self.assertEqual(prediction_ary[0], prediction_dict)

    def test_tree_of_life_classifier_family(self):
        classifier = TreeOfLifeClassifier()
        prediction_ary = classifier.predict(image_path=EXAMPLE_CAT_IMAGE, rank=Rank.FAMILY, k=2)
        self.assertEqual(len(prediction_ary), 2)
        prediction_dict = {
            'file_name': EXAMPLE_CAT_IMAGE,
            'kingdom': 'Animalia',
            'phylum': 'Chordata',
            'class': 'Mammalia',
            'order': 'Carnivora',
            'family': 'Felidae',
            'score': unittest.mock.ANY
        }
        self.assertEqual(prediction_ary[0], prediction_dict)

    def test_custom_labels_classifier(self):
        classifier = CustomLabelsClassifier()
        results = classifier.predict(image_path=EXAMPLE_CAT_IMAGE, cls_ary=['cat', 'dog'])
        self.assertEqual(results, [
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'cat', 'score': unittest.mock.ANY},
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'dog', 'score': unittest.mock.ANY},
        ])
