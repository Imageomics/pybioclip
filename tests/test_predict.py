import unittest
from bioclip.predict import TreeOfLifeClassifier, Rank
from bioclip.predict import CustomLabelsClassifier
import os
import torch


DIRNAME = os.path.dirname(os.path.realpath(__file__))
EXAMPLE_CAT_IMAGE = os.path.join(DIRNAME, "images", "mycat.jpg")
EXAMPLE_CAT_IMAGE2 = os.path.join(DIRNAME, "images", "mycat.png")

class TestPredict(unittest.TestCase):
    def test_tree_of_life_classifier_species_single(self):
        classifier = TreeOfLifeClassifier()
        prediction_ary = classifier.predict(image_paths=EXAMPLE_CAT_IMAGE, rank=Rank.SPECIES)
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

    def test_tree_of_life_classifier_species_ary_one(self):
        classifier = TreeOfLifeClassifier()
        prediction_ary = classifier.predict(image_paths=[EXAMPLE_CAT_IMAGE], rank=Rank.SPECIES)
        self.assertEqual(len(prediction_ary), 5)

    def test_tree_of_life_classifier_species_ary_multiple(self):
        classifier = TreeOfLifeClassifier()
        prediction_ary = classifier.predict(image_paths=[EXAMPLE_CAT_IMAGE, EXAMPLE_CAT_IMAGE2],
                                            rank=Rank.SPECIES)
        self.assertEqual(len(prediction_ary), 10)

    def test_tree_of_life_classifier_family(self):
        classifier = TreeOfLifeClassifier()
        prediction_ary = classifier.predict(image_paths=[EXAMPLE_CAT_IMAGE], rank=Rank.FAMILY, k=2)
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
        classifier = CustomLabelsClassifier(cls_ary=['cat', 'dog'])
        prediction_ary = classifier.predict(image_paths=EXAMPLE_CAT_IMAGE)
        self.assertEqual(prediction_ary, [
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'cat', 'score': unittest.mock.ANY},
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'dog', 'score': unittest.mock.ANY},
        ])

    def test_custom_labels_classifier_ary_one(self):
        classifier = CustomLabelsClassifier(cls_ary=['cat', 'dog'])
        prediction_ary = classifier.predict(image_paths=[EXAMPLE_CAT_IMAGE])
        self.assertEqual(prediction_ary, [
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'cat', 'score': unittest.mock.ANY},
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'dog', 'score': unittest.mock.ANY},
        ])

    def test_custom_labels_classifier_ary_multiple(self):
        classifier = CustomLabelsClassifier(cls_ary=['cat', 'dog'])
        prediction_ary = classifier.predict(image_paths=[EXAMPLE_CAT_IMAGE, EXAMPLE_CAT_IMAGE2])
        self.assertEqual(prediction_ary, [
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'cat', 'score': unittest.mock.ANY},
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'dog', 'score': unittest.mock.ANY},
            {'file_name': EXAMPLE_CAT_IMAGE2, 'classification': 'cat', 'score': unittest.mock.ANY},
            {'file_name': EXAMPLE_CAT_IMAGE2, 'classification': 'dog', 'score': unittest.mock.ANY},
        ])


    def test_predict_with_rgba_image(self):
        # Ensure that the classifier can handle RGBA images
        classifier = TreeOfLifeClassifier()
        prediction_ary = classifier.predict(image_paths=[EXAMPLE_CAT_IMAGE2], rank=Rank.SPECIES)
        self.assertEqual(len(prediction_ary), 5)


class TestEmbed(unittest.TestCase):
    def test_get_image_features(self):
        classifier = TreeOfLifeClassifier(device='cpu')
        self.assertEqual(classifier.model_str, 'hf-hub:imageomics/bioclip')
        features = classifier.create_image_features_for_path(EXAMPLE_CAT_IMAGE, normalize=False)
        self.assertEqual(features.shape, torch.Size([512]))
