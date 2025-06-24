import os
import json
from tqdm import tqdm
import torch
from torchvision import transforms
import open_clip as oc
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import collections
import heapq
import PIL.Image
from huggingface_hub import hf_hub_download
from typing import Union, List
from enum import Enum


TOL10M_HF_DATAFILE_REPO = "imageomics/TreeOfLife-10M"
TOL200M_HF_DATAFILE_REPO = "imageomics/TreeOfLife-200M"
HF_DATAFILE_REPO_TYPE = "dataset"

BIOCLIP_V1_MODEL_STR = "hf-hub:imageomics/bioclip" # TODO
BIOCLIP_V2_MODEL_STR = "hf-hub:imageomics/bioclip-2"
BIOCLIP_MODEL_STR = BIOCLIP_V2_MODEL_STR
TOL_MODELS = {
    BIOCLIP_V1_MODEL_STR: TOL10M_HF_DATAFILE_REPO,
    BIOCLIP_V2_MODEL_STR: TOL200M_HF_DATAFILE_REPO
}
PRED_FILENAME_KEY = "file_name"
PRED_CLASSICATION_KEY = "classification"
PRED_SCORE_KEY = "score"


# Use secure model loading unless explicitly disabled.
# See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more.
if not "TORCH_FORCE_WEIGHTS_ONLY_LOAD" in os.environ:
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "true"


OPENA_AI_IMAGENET_TEMPLATE = [
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a photo of many {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a photo of the hard to see {c}.",
    lambda c: f"a low resolution photo of the {c}.",
    lambda c: f"a rendering of a {c}.",
    lambda c: f"graffiti of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a cropped photo of the {c}.",
    lambda c: f"a tattoo of a {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"a photo of a hard to see {c}.",
    lambda c: f"a bright photo of a {c}.",
    lambda c: f"a photo of a clean {c}.",
    lambda c: f"a photo of a dirty {c}.",
    lambda c: f"a dark photo of the {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"the plastic {c}.",
    lambda c: f"a photo of the cool {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a painting of the {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"a pixelated photo of the {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a bright photo of the {c}.",
    lambda c: f"a cropped photo of a {c}.",
    lambda c: f"a plastic {c}.",
    lambda c: f"a photo of the dirty {c}.",
    lambda c: f"a jpeg corrupted photo of a {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a rendering of the {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a photo of one {c}.",
    lambda c: f"a doodle of a {c}.",
    lambda c: f"a close-up photo of the {c}.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"the origami {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"a sketch of a {c}.",
    lambda c: f"a doodle of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a low resolution photo of a {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a rendition of the {c}.",
    lambda c: f"a photo of the clean {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a rendition of a {c}.",
    lambda c: f"a photo of a nice {c}.",
    lambda c: f"a photo of a weird {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a cartoon {c}.",
    lambda c: f"art of a {c}.",
    lambda c: f"a sketch of the {c}.",
    lambda c: f"a embroidered {c}.",
    lambda c: f"a pixelated photo of a {c}.",
    lambda c: f"itap of the {c}.",
    lambda c: f"a jpeg corrupted photo of the {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a plushie {c}.",
    lambda c: f"a photo of the nice {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the weird {c}.",
    lambda c: f"the cartoon {c}.",
    lambda c: f"art of the {c}.",
    lambda c: f"a drawing of the {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"the plushie {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"itap of a {c}.",
    lambda c: f"graffiti of the {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"itap of my {c}.",
    lambda c: f"a photo of a cool {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a tattoo of the {c}.",
]


def get_tol_repo_id(model_str: str) -> str:
    """
    Returns the repository ID for the TreeOfLife datafile based on the model string.
    Args:
        model_str (str): The model string to check.
    Returns:
        str: The Hugging Face repository ID for the TreeOfLife embeddings.
    """
    repo_id = TOL_MODELS.get(model_str)
    if repo_id is None:
        raise ValueError(f"TreeOfLife predictions are only supported for the following models: {', '.join(TOL_MODELS.keys())}")
    return repo_id


def ensure_tol_supported_model(model_str: str):
    """
    Ensures that the provided model string is one of the supported TreeOfLife models.
    Raises a ValueError if the model is not supported.

    Args:
        model_str (str): The model string to check.

    Raises:
        ValueError: If the model string is not one of the supported TreeOfLife models.
    """
    get_tol_repo_id(model_str)  # This will raise ValueError if the model is not supported


class Rank(Enum):
    """Rank for the Tree of Life classification."""
    KINGDOM = 0
    PHYLUM = 1
    CLASS = 2
    ORDER = 3
    FAMILY = 4
    GENUS = 5
    SPECIES = 6

    def get_label(self):
        return self.name.lower()


def get_rank_labels() -> List[str]:
    """
    Retrieve a list of labels for the items in Rank.
    Returns:
        list: A list of labels corresponding to each rank in the Rank.
    """
    return [rank.get_label() for rank in Rank]


# The datafile of names ('txt_emb_species.json') contains species epithet.
# To create a label for species we concatenate the genus and species epithet.
SPECIES_LABEL = Rank.SPECIES.get_label()
SPECIES_EPITHET_LABEL = "species_epithet"
COMMON_NAME_LABEL = "common_name"


def create_bioclip_tokenizer(model_name="ViT-B-16"):
    return oc.get_tokenizer(model_name=model_name)


preprocess_img = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


class BaseClassifier(nn.Module):
    def __init__(self, model_str: str = BIOCLIP_MODEL_STR, pretrained_str: str | None = None, device: Union[str, torch.device] = 'cpu'):
        """
        Initializes the prediction model.

        Parameters:
            model_str (str): The string identifier for the model to be used (defaults to BIOCLIP_MODEL_STR).
            pretrained_str (str, optional): The string identifier for the pretrained model to be loaded.
            device (Union[str, torch.device]): The device on which the model will be run.
        """
        super().__init__()
        self.device = device
        self.load_pretrained_model(model_str=model_str, pretrained_str=pretrained_str)
        self.recorder = None

    def set_recorder(self, recorder):
        self.recorder = recorder

    def record_event(self, images, **kwargs):
        if self.recorder:
            self.recorder.add_prediction(images, **kwargs)

    def load_pretrained_model(self, model_str: str = BIOCLIP_MODEL_STR, pretrained_str: str | None = None):
        self.model_str = model_str or BIOCLIP_MODEL_STR
        pretrained_tags = oc.list_pretrained_tags_by_model(self.model_str)
        if pretrained_str is None and len(pretrained_tags) > 0:
            if len(pretrained_tags) > 1:
                raise ValueError(f"Multiple pretrained tags available {pretrained_tags}, must provide one")
            pretrained_str = pretrained_tags[0]
        self.pretrained_str = pretrained_str
        model, preprocess = oc.create_model_from_pretrained(self.model_str,
                                                            pretrained=pretrained_str,
                                                            device=self.device,
                                                            return_transform=True)
        self.model = torch.compile(model.to(self.device))
        self.preprocess = preprocess_img if self.model_str in TOL_MODELS else preprocess

    @staticmethod
    def ensure_rgb_image(image: str | PIL.Image.Image) -> PIL.Image.Image:
        if isinstance(image, PIL.Image.Image):
            img = image
        else:
            img = PIL.Image.open(image)
        return img.convert("RGB")

    @staticmethod
    def make_key(image: str | PIL.Image.Image, idx: int) -> str:
        if isinstance(image, PIL.Image.Image):
            return f"{idx}"
        else:
            return image

    @torch.no_grad()
    def create_image_features(self, images: List[PIL.Image.Image], normalize : bool = True) -> torch.Tensor:
        preprocessed_images = []
        for img in images:
            prep_img = self.preprocess(img).to(self.device)
            preprocessed_images.append(prep_img)
        preprocessed_image_tensor = torch.stack(preprocessed_images)
        img_features = self.model.encode_image(preprocessed_image_tensor)
        if normalize:
            return F.normalize(img_features, dim=-1)
        else:
            return img_features

    @torch.no_grad()
    def create_image_features_for_image(self, image: str | PIL.Image.Image, normalize: bool) -> torch.Tensor:
        img = self.ensure_rgb_image(image)
        result = self.create_image_features([img], normalize=normalize)
        return result[0]

    def create_probabilities(self, img_features: torch.Tensor,
                             txt_features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = (self.model.logit_scale.exp() * img_features @ txt_features)
        return F.softmax(logits, dim=1)

    def create_probabilities_for_images(self, images: List[str] | List[PIL.Image.Image],
                                        keys: List[str],
                                        txt_features: torch.Tensor) -> dict[str, torch.Tensor]:
        images = [self.ensure_rgb_image(image) for image in images]
        img_features = self.create_image_features(images)
        probs = self.create_probabilities(img_features, txt_features)
        result = {}
        for i, key in enumerate(keys):
            result[key] = probs[i]
        return result

    def create_batched_probabilities_for_images(self, images: List[str] | List[PIL.Image.Image],
                                                txt_features: torch.Tensor,
                                                batch_size: int | None) -> dict[str, torch.Tensor]:
        if not batch_size:
            batch_size = len(images)
        keys = [self.make_key(image, i) for i,image in enumerate(images)]
        result = {}
        total_images = len(images)
        with tqdm(total=total_images, unit="images") as progress_bar:
            for i in range(0, len(images), batch_size):
                grouped_images = images[i:i + batch_size]
                grouped_keys = keys[i:i + batch_size]
                probs = self.create_probabilities_for_images(grouped_images, grouped_keys, txt_features)
                result.update(probs)
                progress_bar.update(len(grouped_images))
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input tensor representing multiple images, return probabilities for each class for each image.
        Args:
            x (torch.Tensor): Input tensor representing the multiple images.
        Returns:
            torch.Tensor: Softmax probabilities of the logits for each class for each image.
        """
        img_features = self.model.encode_image(x)
        img_features = F.normalize(img_features, dim=-1)
        return self.create_probabilities(img_features, self.txt_embeddings)

    def get_tol_repo_id(self) -> str:
        """
        Returns the repository ID for the TreeOfLife datafile based on the model string.
        Raises:
            ValueError: If the model string is not supported.
        Returns:
            str: The Hugging Face repository ID for the TreeOfLife embeddings.
        """
        return get_tol_repo_id(self.model_str)

    def get_cached_datafile(self, filename: str) -> str:
        """
        Downloads a datafile from the Hugging Face hub and caches it locally.
        Args:
            filename (str): The name of the file to download from the datafile repository.
        Returns:
            str: The local path to the downloaded file.
        """
        return hf_hub_download(repo_id=self.get_tol_repo_id(), filename=filename, repo_type=HF_DATAFILE_REPO_TYPE)

    def get_txt_emb(self) -> torch.Tensor:
        """
        Retrieves TreeOfLife text embeddings for the current model from the associated Hugging Face dataset repo.
        Returns:
            torch.Tensor: A tensor containing the text embeddings for the tree of life.
        """
        txt_emb_npy = self.get_cached_datafile("embeddings/txt_emb_species.npy")
        return torch.from_numpy(np.load(txt_emb_npy))

    def get_txt_names(self) -> List[List[str]]:
        """
        Retrieves TreeOfLife text names for the current model from the  associated Hugging Face dataset repo.
        Returns:
            List[List[str]]: A list of lists, where each inner list contains names corresponding to the text embeddings.
        """
        txt_names_json = self.get_cached_datafile("embeddings/txt_emb_species.json")
        with open(txt_names_json) as fd:
            txt_names = json.load(fd)
        return txt_names


class CustomLabelsClassifier(BaseClassifier):
    """
    A classifier that predicts from a list of custom labels for images.
    """

    def __init__(self, cls_ary: List[str], **kwargs):
        """
        Initializes the classifier with the given class array and additional keyword arguments.

        Parameters:
            cls_ary (List[str]): A list of class names as strings.
        """
        super().__init__(**kwargs)
        self.tokenizer = create_bioclip_tokenizer(self.model_str)
        self.classes = [cls.strip() for cls in cls_ary]
        self.txt_embeddings = self._get_txt_embeddings(self.classes)

    @torch.no_grad()
    def _get_txt_embeddings(self, classnames):
        all_features = []
        for classname in classnames:
            txts = [template(classname) for template in OPENA_AI_IMAGENET_TEMPLATE]
            txts = self.tokenizer(txts).to(self.device)
            txt_features = self.model.encode_text(txts)
            txt_features = F.normalize(txt_features, dim=-1).mean(dim=0)
            txt_features /= txt_features.norm()
            all_features.append(txt_features)
        all_features = torch.stack(all_features, dim=1)
        return all_features

    @torch.no_grad()
    def predict(self, images: List[str] | str | List[PIL.Image.Image], k: int = None,
                batch_size: int = 10) -> dict[str, float]:
        """
        Predicts the probabilities for the given images.

        Parameters:
            images (List[str] | str | List[PIL.Image.Image]): A list of image file paths, a single image file path, or a list of PIL Image objects.
            k (int, optional): The number of top probabilities to return. If not specified or if greater than the number of classes, all probabilities are returned.
            batch_size (int, optional): The number of images to process in a batch.

        Returns:
            List[dict]: A list of dicts with keys "file_name" and the custom class labels.
        """
        if isinstance(images, str):
            images = [images]
        probs = self.create_batched_probabilities_for_images(images, self.txt_embeddings,
                                                             batch_size=batch_size)
        result = []
        for i, image in enumerate(images):
            key = self.make_key(image, i)
            img_probs = probs[key]
            if not k or k > len(self.classes):
                k = len(self.classes)
            result.extend(self.group_probs(key, img_probs, k))

        self.record_event(images=images, k=k, batch_size=batch_size)
        return result

    def group_probs(self, image_key: str, img_probs: torch.Tensor, k: int = None) -> List[dict[str, float]]:
        result = []
        topk = img_probs.topk(k)
        for i, prob in zip(topk.indices, topk.values):
            result.append({
                PRED_FILENAME_KEY: image_key,
                PRED_CLASSICATION_KEY: self.classes[i],
                PRED_SCORE_KEY: prob.item()
            })
        return result


class CustomLabelsBinningClassifier(CustomLabelsClassifier):
    """
    A classifier that creates predictions for images based on custom labels, groups the labels, and calculates probabilities for each group.
    """

    def __init__(self, cls_to_bin: dict, **kwargs):
        """
        Initializes the class with a dictionary mapping class labels to binary values.

        Args:
            cls_to_bin (dict): A dictionary where keys are class labels and values are binary values.
            **kwargs: Additional keyword arguments passed to the superclass initializer.

        Raises:
            ValueError: If any value in `cls_to_bin` is empty, null, or NaN.
        """
        super().__init__(cls_ary=cls_to_bin.keys(), **kwargs)
        self.cls_to_bin = cls_to_bin
        if any([pd.isna(x) or not x for x in cls_to_bin.values()]):
            raise ValueError("Empty, null, or nan are not allowed for bin values.")

    def group_probs(self, image_key: str, img_probs: torch.Tensor, k: int = None) -> List[dict[str, float]]:
        result = []
        output = collections.defaultdict(float)
        for i in range(len(self.classes)):
            name = self.cls_to_bin[self.classes[i]]
            output[name] += img_probs[i]
        topk_names = heapq.nlargest(k, output, key=output.get)
        for name in topk_names:
            result.append({
                PRED_FILENAME_KEY: image_key,
                PRED_CLASSICATION_KEY: name,
                PRED_SCORE_KEY: output[name].item()
            })
        return result


def predict_classifications_from_list(img: Union[PIL.Image.Image, str], cls_ary: List[str], device: Union[str, torch.device] = 'cpu') -> dict[str, float]:
    classifier = CustomLabelsClassifier(cls_ary=cls_ary, device=device)
    return classifier.predict([img])


def get_tol_classification_labels(rank: Rank) -> List[str]:
    names = []
    for i in range(rank.value + 1):
        i_rank = Rank(i)
        if i_rank == Rank.SPECIES:
            names.append(SPECIES_EPITHET_LABEL)
        rank_name = i_rank.name.lower()
        names.append(rank_name)
    if rank == Rank.SPECIES:
        names.append(COMMON_NAME_LABEL)
    return names


def create_classification_dict(names: List[List[str]], rank: Rank) -> dict[str, str]:
    scientific_names = names[0]
    common_name = names[1]
    classification_dict = {}
    for idx, label in enumerate(get_tol_classification_labels(rank=rank)):
        if label == SPECIES_LABEL:
            value = scientific_names[-2] + " " + scientific_names[-1]
        elif label == COMMON_NAME_LABEL:
            value = common_name
        else:
            value = scientific_names[idx]
        classification_dict[label] = value
    return classification_dict


def join_names(classification_dict: dict[str, str]) -> str:
    return " ".join(classification_dict.values())


class TreeOfLifeClassifier(BaseClassifier):
    """
    A classifier for predicting taxonomic ranks for images.
    """
    def __init__(self, **kwargs):
        """
        See `BaseClassifier` for details on `**kwargs`.
        """
        super().__init__(**kwargs)
        self.txt_embeddings = self.get_txt_emb().to(self.device)
        self.txt_names = self.get_txt_names()
        self._subset_txt_embeddings = None
        self._subset_txt_names = None

    def get_txt_embeddings(self):
        if self._subset_txt_embeddings is None:
            return self.txt_embeddings
        return self._subset_txt_embeddings

    def get_current_txt_names(self):
        if self._subset_txt_names is None:
            return self.txt_names
        return self._subset_txt_names

    def get_classification_dict(self, idx: int, rank: Rank):
        name_ary = self.get_current_txt_names()[idx]
        return create_classification_dict(name_ary, rank)

    def get_label_data(self) -> pd.DataFrame:
        """
        Retrieves label data for the tree of life embeddings as a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing label data for TOL embeddings.
        """

        data = []
        for name_ary in self.txt_names:
            data.append(create_classification_dict(names=name_ary, rank=Rank.SPECIES))
        return pd.DataFrame(data, copy=True)
    
    def create_taxa_filter(self, rank: Rank, user_values: List[str]) -> List[bool]:
        """
        Creates a filter for taxa based on the specified rank and user-provided values.
        
        Args:
            rank (Rank): The taxonomic rank to filter by.
            user_values (List[str]): A list of user-provided values to filter the taxa.

        Returns:
            List[bool]: A list of boolean values indicating whether each entry in the 
                        label data matches any of the user-provided values.

        Raises:
            ValueError: If any of the user-provided values are not found in the label data 
                        for the specified taxonomic rank.
        """

        taxa_column = rank.get_label()
        label_data = self.get_label_data()

        # Ensure all user values exist
        pd_user_values = pd.Series(user_values, name=taxa_column)
        unknown_values = pd_user_values[~pd_user_values.isin(label_data[taxa_column])]
        if not unknown_values.empty:
            bad_species = ", ".join(unknown_values.values)
            raise ValueError(f"Unknown {taxa_column} received: {bad_species}. Only known {taxa_column} may be used.")

        return label_data[taxa_column].isin(pd_user_values)

    def create_taxa_filter_from_csv(self, csv_path: str):
        """
        Creates a taxa filter from a CSV file.
        This method reads a CSV file, validates the name of the first column
        against allowed rank labels, and creates a taxa filter based on the
        values in that column.
        Args:
            csv_path (str): The file path to the CSV file.
        Returns:
            A taxa filter object created using the specified rank and values
            from the CSV file.
        Raises:
            ValueError: If the first column name of the CSV file is not one of
            the allowed rank labels.
        """
        df = pd.read_csv(csv_path)
        column_name = str(df.columns[0]).lower()
        allowed_column_names = get_rank_labels()
        if column_name not in allowed_column_names:
            columns = ','.join(allowed_column_names)
            raise ValueError(f"The first column of {csv_path} is named '{column_name}' but must be one of {columns}.")
        filter_rank = Rank[column_name.upper()]
        return self.create_taxa_filter(filter_rank, df[column_name].values)

    def apply_filter(self, keep_labels_ary: List[bool]):
        """
        Filters the TOL embeddings based on the provided boolean array. See `create_taxa_filter()` for an easy way to create this parameter.

        Args:
            keep_labels_ary (List[bool]): A list of boolean values indicating which 
                                          TOL embeddings to keep.

        Raises:
            ValueError: If the length of keep_labels_ary does not match the expected length.
        """

        if len(keep_labels_ary) != len(self.txt_names):
            expected = len(self.txt_names)
            raise ValueError("Invalid keep_embeddings values. " + 
                             f"This parameter should be a list containing {expected} items.")
        embeddings = []
        names = []
        for idx, keep in enumerate(keep_labels_ary):
            if keep:
                embeddings.append(self.txt_embeddings[:,idx])
                names.append(self.txt_names[idx])
        self._subset_txt_embeddings = torch.stack(embeddings, dim=1)
        self._subset_txt_names = names

    def format_species_probs(self, image_key: str, probs: torch.Tensor, k: int = 5) -> List[dict[str, float]]:
        # Prevent error when probs is smaller than k
        k = min(k, probs.shape[0])
        topk = probs.topk(k)
        result = []
        for i, prob in zip(topk.indices, topk.values):
            item = { PRED_FILENAME_KEY: image_key }
            item.update(self.get_classification_dict(i, Rank.SPECIES))
            item[PRED_SCORE_KEY] = prob.item()
            result.append(item)
        return result

    def format_grouped_probs(self, image_key: str, probs: torch.Tensor, rank: Rank, min_prob: float = 1e-9, k: int = 5) -> List[dict[str, float]]:
        output = collections.defaultdict(float)
        class_dict_lookup = {}
        name_to_class_dict = {}
        for i in torch.nonzero(probs > min_prob).squeeze():
            classification_dict = self.get_classification_dict(i, rank)
            name = join_names(classification_dict)
            class_dict_lookup[name] = classification_dict
            output[name] += probs[i]
            name_to_class_dict[name] = classification_dict
        topk_names = heapq.nlargest(k, output, key=output.get)
        prediction_ary = []
        for name in topk_names:
            item = { PRED_FILENAME_KEY: image_key }
            item.update(name_to_class_dict[name])
            item[PRED_SCORE_KEY] = output[name].item()
            prediction_ary.append(item)
        return prediction_ary

    @torch.no_grad()
    def predict(self, images: List[str] | str | List[PIL.Image.Image], rank: Rank, 
                min_prob: float = 1e-9, k: int = 5, batch_size: int = 10) -> dict[str, dict[str, float]]:
        """
        Predicts probabilities for supplied taxa rank for given images using the Tree of Life embeddings.

        Parameters:
            images (List[str] | str | List[PIL.Image.Image]): A list of image file paths, a single image file path, or a list of PIL Image objects.
            rank (Rank): The rank at which to make predictions (e.g., species, genus).
            min_prob (float, optional): The minimum probability threshold for predictions.
            k (int, optional): The number of top predictions to return.
            batch_size (int, optional): The number of images to process in a batch.

        Returns:
            List[dict]: A list of dicts with keys "file_name", taxon ranks, "common_name", and "score".
        """

        if isinstance(images, str):
            images = [images]
        probs = self.create_batched_probabilities_for_images(images, self.get_txt_embeddings(),
                                                             batch_size=batch_size)
        result = []
        for i, image in enumerate(images):
            key = self.make_key(image, i)
            image_probs = probs[key].cpu()
            if rank == Rank.SPECIES:
                result.extend(self.format_species_probs(key, image_probs, k))
            else:
                result.extend(self.format_grouped_probs(key, image_probs, rank, min_prob, k))
        self.record_event(images=images, rank=rank.get_label(), min_prob=min_prob, k=k, batch_size=batch_size)
        return result


def predict_classification(img: Union[PIL.Image.Image, str], rank: Rank, device: Union[str, torch.device] = 'cpu',
                           min_prob: float = 1e-9, k: int = 5) -> dict[str, float]:
    """
    Predicts from the entire tree of life.
    If targeting a higher rank than species, then this function predicts among all
    species, then sums up species-level probabilities for the given rank.
    """
    classifier = TreeOfLifeClassifier(device=device)
    return classifier.predict([img], rank, min_prob, k)
