from enum import Enum
from typing import List

TOL10M_HF_DATAFILE_REPO = "imageomics/TreeOfLife-10M"
TOL200M_HF_DATAFILE_REPO = "imageomics/TreeOfLife-200M"
HF_DATAFILE_REPO_TYPE = "dataset"

BIOCLIP_V1_MODEL_STR = "hf-hub:imageomics/bioclip"  # TODO
BIOCLIP_V2_MODEL_STR = "hf-hub:imageomics/bioclip-2"
BIOCLIP_MODEL_STR = BIOCLIP_V2_MODEL_STR
TOL_MODELS = {
    BIOCLIP_V1_MODEL_STR: TOL10M_HF_DATAFILE_REPO,
    BIOCLIP_V2_MODEL_STR: TOL200M_HF_DATAFILE_REPO
}


DEFAULT_BATCH_SIZE = 10


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
