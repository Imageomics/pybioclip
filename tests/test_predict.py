import unittest
from unittest.mock import patch, mock_open, Mock, ANY
from bioclip.predict import TreeOfLifeClassifier, Rank, get_rank_labels
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

    def test_tree_of_life_classifier_multiple_pil_batching(self):
        classifier = TreeOfLifeClassifier()
        img1 = PIL.Image.open(EXAMPLE_CAT_IMAGE)
        img2 = PIL.Image.open(EXAMPLE_CAT_IMAGE2)

        prediction_ary = classifier.predict(images=[img1, img2],
                                            rank=Rank.SPECIES,
                                            batch_size=1)

        self.assertEqual(len(prediction_ary), 10)
        for i in range(0, 5):
            self.assertEqual(prediction_ary[i]['file_name'], '0')
        for i in range(5, 10):
            self.assertEqual(prediction_ary[i]['file_name'], '1')

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

    def test_tree_of_life_classifier_groups_probs_on_cpu(self):
        # Ensure that the probabilities are moved to the cpu
        # before grouping to avoid performance issues
        classifier = TreeOfLifeClassifier()

        # Have create_probabilities_for_images return mock probs
        # with values returned from cpu()
        probs = Mock()
        probs.cpu.return_value = torch.Tensor([0.1, 0.2, 0.3])
        classifier.create_probabilities_for_images = Mock()
        classifier.create_probabilities_for_images.return_value = {
            EXAMPLE_CAT_IMAGE: probs
        }

        # Mock format_grouped_probs so we can check the parameters
        classifier.format_grouped_probs = Mock()
        classifier.format_grouped_probs.return_value = []

        classifier.predict(images=[EXAMPLE_CAT_IMAGE], rank=Rank.CLASS, k=2)

        # Ensure that the probabilities were moved to the cpu
        classifier.format_grouped_probs.assert_called_with(
            EXAMPLE_CAT_IMAGE,
            probs.cpu.return_value,
            Rank.CLASS,
            ANY, 2
        )

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

    def test_predict_with_batch_size(self):
        classifier = TreeOfLifeClassifier()
        classifier.create_probabilities_for_images = Mock()
        classifier.create_probabilities_for_images.return_value = {
            EXAMPLE_CAT_IMAGE: torch.tensor([1, 0, 0, 0, 0]),
            EXAMPLE_CAT_IMAGE2: torch.tensor([1, 0, 0, 0, 0]),
        }
        prediction_ary = classifier.predict(images=[EXAMPLE_CAT_IMAGE, EXAMPLE_CAT_IMAGE2],
                                            rank=Rank.SPECIES, batch_size=1)
        self.assertEqual(classifier.create_probabilities_for_images.call_count, 2)
        self.assertEqual(len(prediction_ary), 10)

        classifier.create_probabilities_for_images.reset_mock()
        prediction_ary = classifier.predict(images=[EXAMPLE_CAT_IMAGE, EXAMPLE_CAT_IMAGE2],
                                            rank=Rank.SPECIES, batch_size=2)
        self.assertEqual(classifier.create_probabilities_for_images.call_count, 1)

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

    def test_forward(self):
        classifier = TreeOfLifeClassifier()
        img = classifier.ensure_rgb_image(EXAMPLE_CAT_IMAGE)
        img_features = torch.stack([classifier.preprocess(img)])
        result = classifier.forward(x=img_features)
        self.assertEqual(result.shape, torch.Size([1, len(classifier.txt_names)]))

    def test_create_taxa_filter_from_csv(self):
        classifier = TreeOfLifeClassifier()
        csv_content = 'species\nUrsus arctos\nUrsus americanus'
        with patch("builtins.open", mock_open(read_data=csv_content)):
            taxa_filter = classifier.create_taxa_filter_from_csv('somefile.csv')
        self.assertEqual(len(taxa_filter), len(classifier.txt_names))
        # Should have two embeddings since we asked for two species
        self.assertEqual(sum(taxa_filter), 2)

        csv_content = 'species\nUrsus arctos\nUrsus fakename'
        with self.assertRaises(ValueError) as raised_exception:
            with patch("builtins.open", mock_open(read_data=csv_content)):
                taxa_filter = classifier.create_taxa_filter_from_csv('somefile.csv')
        self.assertEqual(str(raised_exception.exception),
                         'Unknown species received: Ursus fakename. Only known species may be used.')

    def test_get_rank_labels(self):
        self.assertEqual(','.join(get_rank_labels()), 'kingdom,phylum,class,order,family,genus,species')

    def test_format_species_probs_too_few_species(self):
        classifier = TreeOfLifeClassifier()

        # test when k < number of probabilities
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        top_probs = classifier.format_species_probs(EXAMPLE_CAT_IMAGE, probs, k=5)
        self.assertEqual(len(top_probs), 5)
        self.assertEqual(top_probs[0]['file_name'], EXAMPLE_CAT_IMAGE)

        # test when k > number of probabilities
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        top_probs = classifier.format_species_probs(EXAMPLE_CAT_IMAGE, probs, k=5)


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
