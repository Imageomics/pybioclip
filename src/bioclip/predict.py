import json
import torch
from torchvision import transforms
from open_clip import create_model, get_tokenizer
import torch.nn.functional as F
import numpy as np
import collections
import heapq
import PIL.Image
from huggingface_hub import hf_hub_download
from typing import Union, List
from enum import Enum


HF_DATAFILE_REPO = "imageomics/bioclip-demo"
HF_DATAFILE_REPO_TYPE = "space"
MODEL_STR = "hf-hub:imageomics/bioclip"
PRED_FILENAME_KEY = "file_name"
PRED_CLASSICATION_KEY = "classification"
PRED_SCORE_KEY = "score"

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


def get_cached_datafile(filename:str):
    return hf_hub_download(repo_id=HF_DATAFILE_REPO, filename=filename, repo_type=HF_DATAFILE_REPO_TYPE)


def get_txt_emb():
    txt_emb_npy = get_cached_datafile("txt_emb_species.npy")
    return torch.from_numpy(np.load(txt_emb_npy))


def get_txt_names():
    txt_names_json = get_cached_datafile("txt_emb_species.json")
    with open(txt_names_json) as fd:
        txt_names = json.load(fd)
    return txt_names


def open_image(image_path):
    img = PIL.Image.open(image_path)
    return img.convert("RGB")


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

class Rank(Enum):
    KINGDOM = 0
    PHYLUM = 1
    CLASS = 2
    ORDER = 3
    FAMILY = 4
    GENUS = 5
    SPECIES = 6

    def get_label(self):
        return self.name.lower()


# The datafile of names ('txt_emb_species.json') contains species epithet.
# To create a label for species we concatenate the genus and species epithet.
SPECIES_LABEL = Rank.SPECIES.get_label()
SPECIES_EPITHET_LABEL = "species_epithet"
COMMON_NAME_LABEL = "common_name"


def create_bioclip_model(model_str, device="cuda"):
    model = create_model(model_str, output_dict=True, require_pretrained=True)
    model = model.to(device)
    return torch.compile(model)


def create_bioclip_tokenizer(tokenizer_str="ViT-B-16"):
    return get_tokenizer(tokenizer_str)


class CustomLabelsClassifier(object):
    def __init__(self, cls_ary: List[str], device: Union[str, torch.device] = 'cpu', model_str: str = MODEL_STR):
        self.device = device
        self.model = create_bioclip_model(device=device, model_str=model_str)
        self.model_str = model_str
        self.tokenizer = create_bioclip_tokenizer()
        self.classes = [cls.strip() for cls in cls_ary]
        self.txt_features = self._get_txt_features(self.classes)

    @torch.no_grad()
    def _get_txt_features(self, classnames):
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
    def predict(self, image_path: str) -> dict[str, float]:
        img = open_image(image_path)

        img = preprocess_img(img).to(self.device)
        img_features = self.model.encode_image(img.unsqueeze(0))
        img_features = F.normalize(img_features, dim=-1)

        logits = (self.model.logit_scale.exp() * img_features @ self.txt_features).squeeze()
        probs = F.softmax(logits, dim=0).to("cpu").tolist()
        pred_list = []
        for cls, prob in zip(self.classes, probs):
            pred_list.append({
                PRED_FILENAME_KEY: image_path,
                PRED_CLASSICATION_KEY: cls,
                PRED_SCORE_KEY: prob
            })
        return pred_list


def predict_classifications_from_list(img: Union[PIL.Image.Image, str], cls_ary: List[str], device: Union[str, torch.device] = 'cpu') -> dict[str, float]:
    classifier = CustomLabelsClassifier(cls_ary=cls_ary, device=device)
    return classifier.predict(img)


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


class TreeOfLifeClassifier(object):
    def __init__(self, device: Union[str, torch.device] = 'cpu', model_str: str = MODEL_STR):
        self.device = device
        self.model = create_bioclip_model(device=device, model_str=model_str)
        self.model_str = model_str
        self.txt_emb = get_txt_emb().to(device)
        self.txt_names = get_txt_names()

    @torch.no_grad()
    def get_image_features(self, image_path: str) -> torch.Tensor:
        img = open_image(image_path)
        return self.encode_image(img)

    def encode_image(self, img: PIL.Image.Image) -> torch.Tensor:
        img = preprocess_img(img).to(self.device)
        img_features = self.model.encode_image(img.unsqueeze(0))
        return img_features

    def predict_species(self, img: PIL.Image.Image) -> torch.Tensor:
        img_features = self.encode_image(img)
        img_features = F.normalize(img_features, dim=-1)
        logits = (self.model.logit_scale.exp() * img_features @ self.txt_emb).squeeze()
        probs = F.softmax(logits, dim=0)
        return probs

    def format_species_probs(self, image_path: str, probs: torch.Tensor, k: int = 5) -> List[dict[str, float]]:
        topk = probs.topk(k)
        result = []
        for i, prob in zip(topk.indices, topk.values):
            item = { PRED_FILENAME_KEY: image_path }
            item.update(create_classification_dict(self.txt_names[i], Rank.SPECIES))
            item[PRED_SCORE_KEY] = prob.item()
            result.append(item)
        return result

    def format_grouped_probs(self, image_path: str, probs: torch.Tensor, rank: Rank, min_prob: float = 1e-9, k: int = 5) -> List[dict[str, float]]:
        output = collections.defaultdict(float)
        class_dict_lookup = {}
        name_to_class_dict = {}
        for i in torch.nonzero(probs > min_prob).squeeze():
            classification_dict = create_classification_dict(self.txt_names[i], rank)
            name = join_names(classification_dict)
            class_dict_lookup[name] = classification_dict
            output[name] += probs[i]
            name_to_class_dict[name] = classification_dict
        topk_names = heapq.nlargest(k, output, key=output.get)
        prediction_ary = []
        for name in topk_names:
            item = { PRED_FILENAME_KEY: image_path }
            item.update(name_to_class_dict[name])
            item[PRED_SCORE_KEY] = output[name].item()
            prediction_ary.append(item)
        return prediction_ary

    @torch.no_grad()
    def predict(self, image_path: str, rank: Rank, min_prob: float = 1e-9, k: int = 5) -> List[dict[str, float]]:
        img = open_image(image_path)
        probs = self.predict_species(img)
        if rank == Rank.SPECIES:
            return self.format_species_probs(image_path, probs, k)
        return self.format_grouped_probs(image_path, probs, rank, min_prob, k)


def predict_classification(img: str, rank: Rank, device: Union[str, torch.device] = 'cpu',
                           min_prob: float = 1e-9, k: int = 5) -> dict[str, float]:
    """
    Predicts from the entire tree of life.
    If targeting a higher rank than species, then this function predicts among all
    species, then sums up species-level probabilities for the given rank.
    """
    classifier = TreeOfLifeClassifier(device=device)
    return classifier.predict(img, rank, min_prob, k)
