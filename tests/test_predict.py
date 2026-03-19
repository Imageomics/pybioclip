import unittest
from unittest.mock import patch, mock_open, Mock, ANY
from bioclip.predict import TreeOfLifeClassifier, BaseClassifier, Rank, get_rank_labels
from bioclip.predict import CustomLabelsClassifier
from bioclip.predict import CustomLabelsBinningClassifier
from bioclip.predict import ensure_tol_supported_model
from bioclip.predict import get_tol_repo_id
from bioclip import BIOCLIP_V2_MODEL_STR, BIOCLIP_V1_MODEL_STR
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
            'species_epithet': unittest.mock.ANY,
            'species': unittest.mock.ANY,
            'common_name': unittest.mock.ANY,
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

    def test_predict_with_callback(self):
        """Test callback is invoked with correct parameters"""
        classifier = TreeOfLifeClassifier()
        callback_calls = []

        def test_callback(processed, total):
            callback_calls.append((processed, total))

        classifier.predict(images=[EXAMPLE_CAT_IMAGE, EXAMPLE_CAT_IMAGE2],
                           rank=Rank.SPECIES, batch_size=1, callback=test_callback)

        # Verify callback was called with correct progression
        self.assertEqual(len(callback_calls), 2)
        self.assertEqual(callback_calls, [(1, 2), (2, 2)])
        # Verify final call shows completion
        self.assertEqual(callback_calls[-1][0], callback_calls[-1][1])

    def test_predict_callback_disables_tqdm(self):
        """Test tqdm is disabled when callback is provided"""
        classifier = CustomLabelsClassifier(cls_ary=['cat', 'dog'])

        with patch('bioclip.predict.tqdm') as mock_tqdm:
            mock_progress = Mock()
            mock_tqdm.return_value.__enter__.return_value = mock_progress
            classifier.predict(images=[EXAMPLE_CAT_IMAGE], callback=lambda p, t: None)

            # Verify tqdm was called with 'disable=True'
            mock_tqdm.assert_called_once()
            call_kwargs = mock_tqdm.call_args[1]
            self.assertTrue(call_kwargs.get('disable'))
            mock_progress.update.assert_not_called()

    def test_predict_without_callback_uses_tqdm(self):
        """Test tqdm is enabled when no callback provided"""
        classifier = CustomLabelsClassifier(cls_ary=['cat', 'dog'])

        with patch('bioclip.predict.tqdm') as mock_tqdm:
            mock_progress = Mock()
            mock_tqdm.return_value.__enter__.return_value = mock_progress
            classifier.predict(images=[EXAMPLE_CAT_IMAGE])

            # Verify tqdm was called with 'disable=False'
            mock_tqdm.assert_called_once()
            call_kwargs = mock_tqdm.call_args[1]
            self.assertFalse(call_kwargs.get('disable'))
            self.assertGreater(mock_progress.update.call_count, 0)

    def test_predict_callback_with_multiple_batches(self):
        """Test callback is invoked correctly with multiple batches"""
        classifier = CustomLabelsClassifier(cls_ary=['cat', 'dog'])
        callback_calls = []

        def test_callback(processed, total):
            callback_calls.append((processed, total))

        # Use 3 images with batch_size=2 to ensure multiple batches
        classifier.predict(images=[EXAMPLE_CAT_IMAGE, EXAMPLE_CAT_IMAGE2, EXAMPLE_CAT_IMAGE],
                           batch_size=2, callback=test_callback)

        # Verify that with 3 images and batch_size=2 (2 batches: [2,1]), the callback is called once per batch (2 calls)
        self.assertEqual(len(callback_calls), 2)
        self.assertEqual(callback_calls[0], (2, 3))
        self.assertEqual(callback_calls[1], (3, 3))

    def test_get_label_data(self):
        classifier = TreeOfLifeClassifier()
        df = classifier.get_label_data()
        self.assertEqual(list(df.columns), ['kingdom', 'phylum', 'class', 'order', 'family', 'genus',
                                            'species_epithet', 'species','common_name'])

    def test_create_taxa_filter(self):
        classifier = TreeOfLifeClassifier()
        taxa_filter = classifier.create_taxa_filter(
            Rank.SPECIES,
            user_values=['Ursus arctos', 'Ursus americanus']
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
        self.assertEqual(classifier.get_txt_embeddings().shape, torch.Size([768, 1]))
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
        self.assertEqual(classifier.model_str, 'hf-hub:imageomics/bioclip-2')
        features = classifier.create_image_features_for_image(EXAMPLE_CAT_IMAGE, normalize=False)
        self.assertEqual(features.shape, torch.Size([768]))

    def test_get_image_features_pil(self):
        classifier = TreeOfLifeClassifier(device='cpu')
        self.assertEqual(classifier.model_str, 'hf-hub:imageomics/bioclip-2')
        features = classifier.create_image_features_for_image(PIL.Image.open(EXAMPLE_CAT_IMAGE), normalize=False)
        self.assertEqual(features.shape, torch.Size([768]))


class TestEnsureTolSupportedModel(unittest.TestCase):
    def test_ensure_tol_supported_model_valid(self):
        # Should not raise for supported models
        try:
            ensure_tol_supported_model(BIOCLIP_V1_MODEL_STR)
            ensure_tol_supported_model(BIOCLIP_V2_MODEL_STR)
        except Exception as e:
            self.fail(f"ensure_tol_supported_model raised unexpectedly: {e}")

    def test_ensure_tol_supported_model_invalid(self):
        with self.assertRaises(ValueError) as cm:
            ensure_tol_supported_model("hf-hub:some/unsupported-model")
        self.assertIn("TreeOfLife predictions are only supported for the following models", str(cm.exception))

    def test_get_tol_repo_id(self):
        # Test for BIOCLIP_V2_MODEL_STR
        self.assertEqual(get_tol_repo_id(BIOCLIP_V2_MODEL_STR), "imageomics/TreeOfLife-200M")

        # Test for BIOCLIP_V1_MODEL_STR
        self.assertEqual(get_tol_repo_id(BIOCLIP_V1_MODEL_STR), "imageomics/TreeOfLife-10M")

        # Test for unsupported model string
        with self.assertRaises(ValueError) as cm:
            get_tol_repo_id("hf-hub:some/unsupported-model")
        self.assertIn("TreeOfLife predictions are only supported for the following models", str(cm.exception))


class TestPredictFromEmbeddings(unittest.TestCase):
    """Tests for predict() with pre-computed image_features."""

    @staticmethod
    def _assert_results_equal_ignoring_file_name(test_case, expected, actual):
        """Assert two result lists are identical except for file_name."""
        test_case.assertEqual(len(expected), len(actual))
        for exp, act in zip(expected, actual):
            for key in exp:
                if key == 'file_name':
                    continue
                test_case.assertEqual(exp[key], act[key],
                    f"Mismatch on key '{key}': {exp[key]} != {act[key]}")

    def test_tol_predict_species_from_features_matches_images(self):
        """Species-level predict from embeddings must match predict from images exactly."""
        classifier = TreeOfLifeClassifier()
        features = classifier.create_image_features(
            [classifier.ensure_rgb_image(EXAMPLE_CAT_IMAGE)]
        )
        result_from_images = classifier.predict(images=EXAMPLE_CAT_IMAGE, rank=Rank.SPECIES, k=5)
        result_from_features = classifier.predict(image_features=features, rank=Rank.SPECIES, k=5)
        self._assert_results_equal_ignoring_file_name(self, result_from_images, result_from_features)
        # file_name should be numeric index when no images provided
        for entry in result_from_features:
            self.assertEqual(entry['file_name'], '0')

    def test_tol_predict_family_from_features_matches_images(self):
        """Family-level predict from embeddings must match predict from images exactly."""
        classifier = TreeOfLifeClassifier()
        features = classifier.create_image_features(
            [classifier.ensure_rgb_image(EXAMPLE_CAT_IMAGE)]
        )
        result_from_images = classifier.predict(images=EXAMPLE_CAT_IMAGE, rank=Rank.FAMILY, k=2)
        result_from_features = classifier.predict(image_features=features, rank=Rank.FAMILY, k=2)
        self._assert_results_equal_ignoring_file_name(self, result_from_images, result_from_features)

    def test_tol_predict_multiple_from_features_matches_images(self):
        """Multi-image predict from embeddings must match predict from images exactly."""
        classifier = TreeOfLifeClassifier()
        img1 = classifier.ensure_rgb_image(EXAMPLE_CAT_IMAGE)
        img2 = classifier.ensure_rgb_image(EXAMPLE_CAT_IMAGE2)
        features = classifier.create_image_features([img1, img2])
        result_from_images = classifier.predict(
            images=[EXAMPLE_CAT_IMAGE, EXAMPLE_CAT_IMAGE2], rank=Rank.SPECIES, k=5
        )
        result_from_features = classifier.predict(image_features=features, rank=Rank.SPECIES, k=5)
        self._assert_results_equal_ignoring_file_name(self, result_from_images, result_from_features)
        # Verify numeric keys for each image's results
        for i in range(5):
            self.assertEqual(result_from_features[i]['file_name'], '0')
        for i in range(5, 10):
            self.assertEqual(result_from_features[i]['file_name'], '1')

    def test_tol_predict_unnormalized_features_matches_images(self):
        """Unnormalized features should be auto-normalized and produce correct results."""
        classifier = TreeOfLifeClassifier()
        # Get unnormalized features
        unnorm_features = classifier.create_image_features(
            [classifier.ensure_rgb_image(EXAMPLE_CAT_IMAGE)], normalize=False
        )
        # Verify they are indeed not normalized
        norms = unnorm_features.norm(dim=-1)
        self.assertFalse(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
        # Predict from unnormalized features
        result_from_images = classifier.predict(images=EXAMPLE_CAT_IMAGE, rank=Rank.SPECIES, k=5)
        result_from_features = classifier.predict(image_features=unnorm_features, rank=Rank.SPECIES, k=5)
        # Classifications should match; scores may have minor float drift from normalization
        self.assertEqual(len(result_from_images), len(result_from_features))
        for img_res, feat_res in zip(result_from_images, result_from_features):
            for key in img_res:
                if key in ('file_name', 'score'):
                    continue
                self.assertEqual(img_res[key], feat_res[key],
                    f"Mismatch on key '{key}': {img_res[key]} != {feat_res[key]}")

    def test_custom_predict_from_features_matches_images(self):
        """CustomLabelsClassifier predict from embeddings must match predict from images exactly."""
        classifier = CustomLabelsClassifier(cls_ary=['cat', 'dog'])
        features = classifier.create_image_features(
            [classifier.ensure_rgb_image(EXAMPLE_CAT_IMAGE)]
        )
        result_from_images = classifier.predict(images=EXAMPLE_CAT_IMAGE)
        result_from_features = classifier.predict(image_features=features)
        self._assert_results_equal_ignoring_file_name(self, result_from_images, result_from_features)
        for entry in result_from_features:
            self.assertEqual(entry['file_name'], '0')

    def test_binning_predict_from_features_matches_images(self):
        """CustomLabelsBinningClassifier predict from embeddings must match predict from images."""
        classifier = CustomLabelsBinningClassifier(cls_to_bin={
            'cat': 'one',
            'mouse': 'two',
            'fish': 'two',
        })
        features = classifier.create_image_features(
            [classifier.ensure_rgb_image(EXAMPLE_CAT_IMAGE)]
        )
        result_from_images = classifier.predict(images=EXAMPLE_CAT_IMAGE)
        result_from_features = classifier.predict(image_features=features)
        self._assert_results_equal_ignoring_file_name(self, result_from_images, result_from_features)

    def test_predict_no_images_no_features_raises(self):
        """Should raise ValueError when neither images nor image_features provided."""
        classifier = TreeOfLifeClassifier()
        with self.assertRaises(ValueError) as cm:
            classifier.predict(rank=Rank.SPECIES)
        self.assertIn("Either images or image_features must be provided", str(cm.exception))

        cls_classifier = CustomLabelsClassifier(cls_ary=['cat', 'dog'])
        with self.assertRaises(ValueError) as cm:
            cls_classifier.predict()
        self.assertIn("Either images or image_features must be provided", str(cm.exception))

    def test_predict_image_features_wrong_dim_raises(self):
        """Should raise ValueError for non-2D image_features tensor."""
        classifier = TreeOfLifeClassifier()
        with self.assertRaises(ValueError) as cm:
            classifier.predict(image_features=torch.randn(768), rank=Rank.SPECIES)
        self.assertIn("2D tensor", str(cm.exception))

    def test_predict_image_features_wrong_embedding_dim_raises(self):
        """Should raise ValueError when embedding_dim doesn't match model's expected dimension."""
        classifier = TreeOfLifeClassifier()
        # Model expects 768 for ViT-L/14, pass 512
        features = torch.randn(1, 512)
        with self.assertRaises(ValueError) as cm:
            classifier.predict(image_features=features, rank=Rank.SPECIES)
        self.assertIn("does not match", str(cm.exception))

    def test_predict_image_features_length_mismatch_raises(self):
        """Should raise ValueError when images and image_features lengths don't match."""
        classifier = TreeOfLifeClassifier()
        features = torch.randn(2, 768)
        with self.assertRaises(ValueError) as cm:
            classifier.predict(
                images=[EXAMPLE_CAT_IMAGE], rank=Rank.SPECIES, image_features=features
            )
        self.assertIn("must match", str(cm.exception))
