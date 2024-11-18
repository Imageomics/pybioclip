import unittest
from bioclip.predict import TreeOfLifeClassifier, Rank
from bioclip.predict import CustomLabelsClassifier
from bioclip.predict import CustomLabelsBinningClassifier
import os
import torch
import pandas as pd
import PIL.Image


DIRNAME = os.path.dirname(os.path.realpath(__file__))
EXAMPLE_CAT_IMAGE = os.path.join(DIRNAME, "images", "mycat.jpg")
EXAMPLE_CAT_IMAGE2 = os.path.join(DIRNAME, "images", "mycat.png")

class TestPredict(unittest.TestCase):
    def test_tree_of_life_classifier_species_single(self):
        classifier = TreeOfLifeClassifier()
        prediction_ary = classifier.predict(images=EXAMPLE_CAT_IMAGE, rank=Rank.SPECIES)
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
        prediction_ary = classifier.predict(images=[EXAMPLE_CAT_IMAGE], rank=Rank.SPECIES)
        self.assertEqual(len(prediction_ary), 5)

    def test_tree_of_life_classifier_species_ary_multiple(self):
        classifier = TreeOfLifeClassifier()
        prediction_ary = classifier.predict(images=[EXAMPLE_CAT_IMAGE, EXAMPLE_CAT_IMAGE2],
                                            rank=Rank.SPECIES)
        self.assertEqual(len(prediction_ary), 10)

    def test_tree_of_life_classifier_species_ary_multiple_pil(self):
        classifier = TreeOfLifeClassifier()
        img1 = PIL.Image.open(EXAMPLE_CAT_IMAGE)
        img2 = PIL.Image.open(EXAMPLE_CAT_IMAGE2)
        prediction_ary = classifier.predict(images=[img1, img2],
                                            rank=Rank.SPECIES)
        self.assertEqual(len(prediction_ary), 10)

    def test_tree_of_life_classifier_family(self):
        classifier = TreeOfLifeClassifier()
        prediction_ary = classifier.predict(images=[EXAMPLE_CAT_IMAGE], rank=Rank.FAMILY, k=2)
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
        prediction_ary = classifier.predict(images=EXAMPLE_CAT_IMAGE)
        self.assertEqual(prediction_ary, [
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'cat', 'score': unittest.mock.ANY},
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'dog', 'score': unittest.mock.ANY},
        ])

    def test_custom_labels_classifier_ary_one(self):
        classifier = CustomLabelsClassifier(cls_ary=['cat', 'dog'])
        prediction_ary = classifier.predict(images=[EXAMPLE_CAT_IMAGE])
        self.assertEqual(prediction_ary, [
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'cat', 'score': unittest.mock.ANY},
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'dog', 'score': unittest.mock.ANY},
        ])

    def test_custom_labels_classifier_ary_multiple(self):
        classifier = CustomLabelsClassifier(cls_ary=['cat', 'dog'])
        prediction_ary = classifier.predict(images=[EXAMPLE_CAT_IMAGE, EXAMPLE_CAT_IMAGE2])
        self.assertEqual(prediction_ary, [
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'cat', 'score': unittest.mock.ANY},
            {'file_name': EXAMPLE_CAT_IMAGE, 'classification': 'dog', 'score': unittest.mock.ANY},
            {'file_name': EXAMPLE_CAT_IMAGE2, 'classification': 'cat', 'score': unittest.mock.ANY},
            {'file_name': EXAMPLE_CAT_IMAGE2, 'classification': 'dog', 'score': unittest.mock.ANY},
        ])

    def test_custom_labels_classifier_ary_multiple_pil(self):
        classifier = CustomLabelsClassifier(cls_ary=['cat', 'dog'])
        img1 = PIL.Image.open(EXAMPLE_CAT_IMAGE)
        img2 = PIL.Image.open(EXAMPLE_CAT_IMAGE2)
        prediction_ary = classifier.predict(images=[img1, img2])
        self.assertEqual(prediction_ary, [
            {'file_name': '0', 'classification': 'cat', 'score': unittest.mock.ANY},
            {'file_name': '0', 'classification': 'dog', 'score': unittest.mock.ANY},
            {'file_name': '1', 'classification': 'cat', 'score': unittest.mock.ANY},
            {'file_name': '1', 'classification': 'dog', 'score': unittest.mock.ANY},
        ])

    def test_predict_with_rgba_image(self):
        # Ensure that the classifier can handle RGBA images
        classifier = TreeOfLifeClassifier()
        prediction_ary = classifier.predict(images=[EXAMPLE_CAT_IMAGE2], rank=Rank.SPECIES)
        self.assertEqual(len(prediction_ary), 5)

    def test_predict_with_bins(self):
        classifier = CustomLabelsBinningClassifier(cls_to_bin={
            'cat': 'one',
            'mouse': 'two',
            'fish': 'two',
        })
        prediction_ary = classifier.predict(images=[EXAMPLE_CAT_IMAGE2])
        self.assertEqual(len(prediction_ary), 2)
        self.assertEqual(prediction_ary[0]['file_name'], EXAMPLE_CAT_IMAGE2)
        names = set([pred['classification'] for pred in prediction_ary])
        self.assertEqual(names, set(['one', 'two']))

        classifier = CustomLabelsBinningClassifier(cls_to_bin={
            'cat': 'one',
            'mouse': 'two',
            'fish': 'three',
        })
        prediction_ary = classifier.predict(images=[EXAMPLE_CAT_IMAGE2])
        self.assertEqual(len(prediction_ary), 3)
        self.assertEqual(prediction_ary[0]['file_name'], EXAMPLE_CAT_IMAGE2)
        names = set([pred['classification'] for pred in prediction_ary])
        self.assertEqual(names, set(['one', 'two', 'three']))

    def test_predict_with_bins_bad_values(self):
        with self.assertRaises(ValueError) as raised_exceptions:
            CustomLabelsBinningClassifier(cls_to_bin={
                'cat': 'one',
                'mouse': '',
                'fish': 'two',
            })
        self.assertEqual(str(raised_exceptions.exception),
                         "Empty, null, or nan are not allowed for bin values.")
        with self.assertRaises(ValueError) as raised_exceptions:
            CustomLabelsBinningClassifier(cls_to_bin={
                'cat': 'one',
                'mouse': None,
                'fish': 'two',
            })
        self.assertEqual(str(raised_exceptions.exception),
                         "Empty, null, or nan are not allowed for bin values.")
        with self.assertRaises(ValueError) as raised_exceptions:
            CustomLabelsBinningClassifier(cls_to_bin={
                'cat': 'one',
                'mouse': pd.NA,
                'fish': 'two',
            })
        self.assertEqual(str(raised_exceptions.exception),
                         "Empty, null, or nan are not allowed for bin values.")

    def test_predict_with_bins_pil(self):
        classifier = CustomLabelsBinningClassifier(cls_to_bin={
            'cat': 'one',
            'mouse': 'two',
            'fish': 'two',
        })
        img1 = PIL.Image.open(EXAMPLE_CAT_IMAGE)
        prediction_ary = classifier.predict(images=[img1])
        self.assertEqual(len(prediction_ary), 2)
        self.assertEqual(prediction_ary[0]['file_name'], '0')
        names = set([pred['classification'] for pred in prediction_ary])
        self.assertEqual(names, set(['one', 'two']))

    def test_get_label_data(self):
        classifier = TreeOfLifeClassifier()
        df = classifier.get_label_data()
        self.assertEqual(list(df.columns), ['kingdom', 'phylum', 'class', 'order', 'family', 'genus',
                                            'species_epithet', 'species','common_name'])

    def test_create_taxa_filter(self):
        classifier = TreeOfLifeClassifier()
        taxa_filter = classifier.create_taxa_filter(
            Rank.SPECIES,
            user_values=['Ursus arctos', 'Ursus arctos bruinosus']
        )
        self.assertEqual(len(taxa_filter), len(classifier.txt_names))
        # Should have two embeddings since we asked for two species
        self.assertEqual(sum(taxa_filter), 2)

        # Should raise exception for bogus taxa values
        with self.assertRaises(ValueError) as raised_ex:
            classifier.create_taxa_filter(
                Rank.SPECIES,
                user_values=['Ursus arctos', 'Ursus fakespeciesname']
            )
        self.assertEqual(str(raised_ex.exception),
                         "Unknown species received: Ursus fakespeciesname. Only known species may be used.")

    def test_apply_filter(self):
        classifier = TreeOfLifeClassifier()
        # Ensure an error is raised if there are too few boolean values in the array
        with self.assertRaises(ValueError) as raised_ex:
            classifier.apply_filter([False, True])

        # Test that if we filter a single species we get one embedding
        taxon_filter = classifier.get_label_data().species == 'Ursus arctos'
        classifier.apply_filter(taxon_filter)
        self.assertEqual(classifier.get_txt_embeddings().shape, torch.Size([512, 1]))
        self.assertEqual(len(classifier.get_current_txt_names()), 1)


class TestEmbed(unittest.TestCase):
    def test_get_image_features(self):
        classifier = TreeOfLifeClassifier(device='cpu')
        self.assertEqual(classifier.model_str, 'hf-hub:imageomics/bioclip')
        features = classifier.create_image_features_for_image(EXAMPLE_CAT_IMAGE, normalize=False)
        self.assertEqual(features.shape, torch.Size([512]))

    def test_get_image_features_pil(self):
        classifier = TreeOfLifeClassifier(device='cpu')
        self.assertEqual(classifier.model_str, 'hf-hub:imageomics/bioclip')
        features = classifier.create_image_features_for_image(PIL.Image.open(EXAMPLE_CAT_IMAGE), normalize=False)
        self.assertEqual(features.shape, torch.Size([512]))
