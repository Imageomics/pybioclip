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


class Rank(Enum):
    KINGDOM = 0
    PHYLUM = 1
    CLASS = 2
    ORDER = 3
    FAMILY = 4
    GENUS = 5
    SPECIES = 6


def create_bioclip_model(model_str="hf-hub:imageomics/bioclip", device="cuda"):
    model = create_model(model_str, output_dict=True, require_pretrained=True)
    model = model.to(device)
    return torch.compile(model)


def create_bioclip_tokenizer(tokenizer_str="ViT-B-16"):
    return get_tokenizer(tokenizer_str)


@torch.no_grad()
def get_txt_features(classnames, templates, tokenizer, model, device):
    all_features = []
    for classname in classnames:
        txts = [template(classname) for template in templates]
        txts = tokenizer(txts).to(device)
        txt_features = model.encode_text(txts)
        txt_features = F.normalize(txt_features, dim=-1).mean(dim=0)
        txt_features /= txt_features.norm()
        all_features.append(txt_features)
    all_features = torch.stack(all_features, dim=1)
    return all_features


@torch.no_grad()
def predict_classifications_from_list(img: Union[PIL.Image.Image, str], cls_ary: List[str], device: Union[str, torch.device] = 'cpu') -> dict[str, float]:
    if isinstance(img,str):
       img = PIL.Image.open(img)
    model = create_bioclip_model(device=device)
    tokenizer = create_bioclip_tokenizer()
    
    classes = [cls.strip() for cls in cls_ary]
    txt_features = get_txt_features(classes, OPENA_AI_IMAGENET_TEMPLATE, tokenizer=tokenizer, model=model, device=device)

    img = preprocess_img(img).to(device)
    img_features = model.encode_image(img.unsqueeze(0))
    img_features = F.normalize(img_features, dim=-1)

    logits = (model.logit_scale.exp() * img_features @ txt_features).squeeze()
    probs = F.softmax(logits, dim=0).to("cpu").tolist()
    return {cls: prob for cls, prob in zip(classes, probs)}


def format_name(taxon, common):
    taxon = " ".join(taxon)
    if not common:
        return taxon
    return f"{taxon} ({common})"


@torch.no_grad()
def predict_classification(img: Union[PIL.Image.Image, str], rank: Rank, device: Union[str, torch.device] = 'cpu',
                               min_prob: float = 1e-9, k: int = 5) -> dict[str, float]:
    """
    Predicts from the entire tree of life.
    If targeting a higher rank than species, then this function predicts among all
    species, then sums up species-level probabilities for the given rank.
    """
    if isinstance(img,str):
       img = PIL.Image.open(img)
    model = create_bioclip_model(device=device)
    img = preprocess_img(img).to(device)
    txt_emb = get_txt_emb().to(device)
    txt_names = get_txt_names()
    img_features = model.encode_image(img.unsqueeze(0))
    img_features = F.normalize(img_features, dim=-1)

    logits = (model.logit_scale.exp() * img_features @ txt_emb).squeeze()
    probs = F.softmax(logits, dim=0)

    # If predicting species, no need to sum probabilities.
    if rank == Rank.SPECIES:
        topk = probs.topk(k)
        return {
            format_name(*txt_names[i]): prob.item() for i, prob in zip(topk.indices, topk.values)
        }
    # Sum up by the rank
    output = collections.defaultdict(float)
    for i in torch.nonzero(probs > min_prob).squeeze():
        output[" ".join(txt_names[i][0][: rank.value + 1])] += probs[i]

    topk_names = heapq.nlargest(k, output, key=output.get)

    return {name: output[name].item() for name in topk_names}

