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
from bioclip.templates import OPENA_AI_IMAGENET_TEMPLATE
import requests
from io import BytesIO


HF_DATAFILE_REPO = "imageomics/bioclip-demo"
HF_DATAFILE_REPO_TYPE = "space"
PRED_CLASSICATION_KEY = "classification"
PRED_SCORE_KEY = "score"


def create_image(img: str) -> PIL.Image.Image:
    if img.startswith("http://") or img.startswith("https://"):
        response = requests.get(img)
        image = PIL.Image.open(BytesIO(response.content))
    else:
        image = PIL.Image.open(img)
    return image


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
# Labels are needed to account for the species epithet. The datafile of names 'txt_emb_species.json'
# contains species epithet instead of species. To create species we concatenate the genus
# and species epithet.
SPECIES_LABEL = Rank.SPECIES.get_label()
SPECIES_EPITHET_LABEL = "species_epithet"
COMMON_NAME_LABEL = "common_name"


def create_bioclip_model(model_str="hf-hub:imageomics/bioclip", device="cuda"):
    model = create_model(model_str, output_dict=True, require_pretrained=True)
    model = model.to(device)
    return torch.compile(model)


def create_bioclip_tokenizer(tokenizer_str="ViT-B-16"):
    return get_tokenizer(tokenizer_str)


class CustomLabelsClassifier(object):
    def __init__(self, device: Union[str, torch.device] = 'cpu'):
        self.device = device
        self.model = create_bioclip_model(device=device)
        self.tokenizer = create_bioclip_tokenizer()

    @torch.no_grad()
    def get_txt_features(self, classnames):
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

    def predict(self, image_path: str, cls_ary: List[str]) -> dict[str, float]:
        img = create_image(image_path)
        classes = [cls.strip() for cls in cls_ary]
        txt_features = self.get_txt_features(classes)

        img = preprocess_img(img).to(self.device)
        img_features = self.model.encode_image(img.unsqueeze(0))
        img_features = F.normalize(img_features, dim=-1)

        logits = (self.model.logit_scale.exp() * img_features @ txt_features).squeeze()
        probs = F.softmax(logits, dim=0).to("cpu").tolist()
        pred_list = []
        for cls, prob in zip(classes, probs):
            pred_list.append({
                PRED_CLASSICATION_KEY: cls,
                PRED_SCORE_KEY: prob
            })
        return pred_list


@torch.no_grad()
def predict_classifications_from_list(img: Union[PIL.Image.Image, str], cls_ary: List[str], device: Union[str, torch.device] = 'cpu') -> dict[str, float]:
    classifier = CustomLabelsClassifier(device=device)
    return classifier.predict(img, cls_ary)


def format_name(taxon, common):
    taxon = " ".join(taxon)
    if not common:
        return taxon
    return f"{taxon} ({common})"


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
    def __init__(self, device: Union[str, torch.device] = 'cpu'):
        self.device = device
        self.model = create_bioclip_model(device=device)
        self.txt_emb = get_txt_emb()
        self.txt_names = get_txt_names()

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
    
    def format_species_probs(self, probs: torch.Tensor, k: int = 5) -> List[dict[str, float]]:
        topk = probs.topk(k)
        result = []
        for i, prob in zip(topk.indices, topk.values):
            item = {
                PRED_CLASSICATION_KEY: create_classification_dict(self.txt_names[i], Rank.SPECIES),
                PRED_SCORE_KEY: prob.item(),
            }
            result.append(item)
        return result

    def format_grouped_probs(self, probs: torch.Tensor, rank: Rank, min_prob: float = 1e-9, k: int = 5) -> List[dict[str, float]]:
        output = collections.defaultdict(float)
        class_dict_lookup = {}
        for i in torch.nonzero(probs > min_prob).squeeze():
            classification_dict = create_classification_dict(self.txt_names[i], rank)
            name = join_names(classification_dict)
            class_dict_lookup[name] = classification_dict
            output[name] += probs[i]
        topk_names = heapq.nlargest(k, output, key=output.get)
        score_ary = []
        for name in topk_names:
            score_ary.append({
                PRED_CLASSICATION_KEY: class_dict_lookup[name],
                PRED_SCORE_KEY: output[name].item(),
            })
        return score_ary
    
    def predict(self, image_path: str, rank: Rank, min_prob: float = 1e-9, k: int = 5) -> List[dict[str, float]]:
        img = create_image(image_path)
        probs = self.predict_species(img)
        if rank == Rank.SPECIES:
            return self.format_species_probs(probs, k)
        return self.format_grouped_probs(probs, rank, min_prob, k)


@torch.no_grad()
def predict_classification(img: str, rank: Rank, device: Union[str, torch.device] = 'cpu',
                           min_prob: float = 1e-9, k: int = 5) -> dict[str, float]:
    """
    Predicts from the entire tree of life.
    If targeting a higher rank than species, then this function predicts among all
    species, then sums up species-level probabilities for the given rank.
    """
    classifier = TreeOfLifeClassifier(device=device)
    return classifier.predict(img, rank, min_prob, k)
