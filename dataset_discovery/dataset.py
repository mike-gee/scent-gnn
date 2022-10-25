import logging
import random
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import periodictable
import torch
from pysmiles import read_smiles
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from consts import atomic_properties

# filter out pysmiles stereochemical information warnings
logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


def _get_node_data(node: dict, features: List = atomic_properties) -> List:
    """Get the data coressponding to some element in a molecule.

    Args:
        node: dictionary containing information pertaining to element
            in current molecule.
        features: list of features to extract for each element in molecule.

    Returns:
        list of features & metadata associated with element in molecule.
    """
    atomic_data = getattr(periodictable, node["element"])

    assert getattr(atomic_data, "K_beta1_units") == "angstrom"
    assert getattr(atomic_data, "K_alpha_units") == "angstrom"
    assert getattr(atomic_data, "covalent_radius_units") == "angstrom"

    atomic_num = getattr(atomic_data, "number")
    atomic_one_hot = [0 for x in range(118)]
    atomic_one_hot[atomic_num - 1] = 1

    metadata = [
        getattr(atomic_data, x) if x in dir(atomic_data) else 0 for x in features
    ] + [int(node["aromatic"]), node["hcount"]]

    return atomic_one_hot + metadata


def construct_dataset(
    behaviors: pd.DataFrame,
    test_size: float = 0.1,
    features: List = atomic_properties,
) -> Tuple[List, List]:
    """Construct dataset extracting features for molecule.

    Args:
        behaviors: Pandas DataFrame indexed by the SMILES
            encoding of molecule, containing its scents.
        test_size: pct of dataset to use as test set.
        features: list of features to extract for each element in molecule.

    Returns:
        (train_dataset, test_dataset): list of torch_geometric Data classes
            for train and test datasets.

    Notes:
        Ignores stereomolecule information in molecule.

    """
    random.seed(888)
    dataset = []
    for i, row in behaviors.iterrows():
        mol = read_smiles(i)
        edge_index = torch.tensor(
            [[x[0] for x in mol.edges], [x[1] for x in mol.edges]],
            dtype=torch.long,
        )
        X = torch.tensor(
            [_get_node_data(mol.nodes[i], features) for i in mol.nodes],
            dtype=torch.float,
        )
        y = torch.Tensor(row.to_numpy()).reshape(1, -1)
        data = Data(edge_index=edge_index, x=X, y=y)
        dataset.append(data)
    random.shuffle(dataset)
    test_n = int(np.floor(len(dataset) * test_size))
    return dataset[test_n:], dataset[:test_n]


def get_balanced_sampler(dataset: List[Data]) -> WeightedRandomSampler:
    """Construct a WeightedRandomSampler, given some dataset.

    Args:
        dataset: Torch geometric dataset.

    Returns:
        WeighetdRandomSampler for dataset, weighetd by frequency of dataset.
    """
    y_list = [str(data.y.tolist()) for data in dataset]
    y_unique, y_counts = np.unique(y_list, return_counts=True)
    y_counts = 1 / np.array(y_counts)
    freq = {x: y for x, y in zip(y_unique, y_counts)}
    freq_list = list(map(lambda x: freq[x], list(y_list)))
    return WeightedRandomSampler(freq_list, num_samples=len(freq_list))
