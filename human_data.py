import os
import pickle
from collections import OrderedDict
import random
import glob

import pandas as pd
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import dgl
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx
from Bio.PDB import *
import deepchem
import pickle

random.seed(0)

pk = deepchem.dock.ConvexHullPocketFinder()


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    # print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])  # (10, 6, 5, 6, 1) --> total 28


def get_atom_feature(m):
    H = []
    for i in range(len(m)):
        H.append(atom_feature(m[i][0]))
    H = np.array(H)

    return H


def process_protein(pdb_file):
    m = Chem.MolFromPDBFile(pdb_file)
    am = GetAdjacencyMatrix(m)
    pockets = pk.find_pockets(pdb_file)
    n2 = m.GetNumAtoms()
    c2 = m.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    binding_parts = []
    not_in_binding = [i for i in range(0, n2)]
    constructed_graphs = []
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        binding_parts_atoms = []
        idxs = []
        for idx, atom_cord in enumerate(d2):
            if x_min < atom_cord[0] < x_max and y_min < atom_cord[1] < y_max and z_min < atom_cord[2] < z_max:
                binding_parts_atoms.append((m.GetAtoms()[idx], atom_cord))
                idxs.append(idx)
                if idx in not_in_binding:
                    not_in_binding.remove(idx)

        ami = am[np.array(idxs)[:, None], np.array(idxs)]
        H = get_atom_feature(binding_parts_atoms)
        g = nx.convert_matrix.from_numpy_matrix(ami)
        graph = dgl.DGLGraph(g)
        graph.ndata['h'] = torch.Tensor(H)
        graph = dgl.add_self_loop(graph)
        constructed_graphs.append(graph)
        binding_parts.append(binding_parts_atoms)

    constructed_graphs = dgl.batch(constructed_graphs)

    # amj = am[np.array(not_in_binding)[:, None], np.array(not_in_binding)]
    # not_binding_atoms = []
    # for item in not_in_binding:
    #     not_binding_atoms.append((m.GetAtoms()[item], d2[item]))
    # H = get_atom_feature(not_binding_atoms)
    # g = nx.convert_matrix.from_numpy_matrix(amj)
    # graph = dgl.DGLGraph(g)
    # graph.ndata['h'] = torch.Tensor(H)
    # graph = dgl.add_self_loop(graph)
    # constructed_graphs = dgl.batch([constructed_graphs, graph])
    return binding_parts, not_in_binding, constructed_graphs


node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')

zero = np.eye(2)[1]
one = np.eye(2)[0]

df = pd.read_csv("humanSeqPdb.txt")
print(len(df['pdb_id'].unique()))

with open("data.txt", 'r') as fp:
    train_raw = fp.read()
train_set = []

previous_pdb = ""
constructed_graphs = ""
raw_data = train_raw.split("\n")
random.shuffle(raw_data)
raw_data_train = raw_data[0: int(len(raw_data)*0.8)]
raw_data_valid = raw_data[int(len(raw_data)*0.8): int(len(raw_data)*0.9)]
raw_data_test = raw_data[int(len(raw_data)*0.9): int(len(raw_data))]
del raw_data
i = 1

for item in raw_data_train:
    print(i)
    i += 1
    try:
        a = item.split()
        smile = a[0]
        sequence = a[1]
        pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()[:-1]
        if pdb_code != "6g5i" and pdb_code != "5t0j" and pdb_code != "5wve":
            if previous_pdb != pdb_code:
                pdbl = PDBList()
                pdbl.retrieve_pdb_file(
                    pdb_code, pdir='./pdbs/', overwrite=True, file_format="pdb"
                )
                # Rename file to .pdb from .ent
                os.rename(
                    './pdbs/' + "pdb" + pdb_code + ".ent", './pdbs/' + pdb_code + ".pdb"
                )
                # Assert file has been downloaded
                assert any(pdb_code in s for s in os.listdir('./pdbs/'))
                #print(f"Downloaded PDB file for: {pdb_code}")
                _, _, constructed_graphs = process_protein(f"./pdbs/{pdb_code}.pdb")

                previous_pdb = pdb_code

            g = smiles_to_bigraph(smile, node_featurizer=node_featurizer)
            g = dgl.add_self_loop(g)
            if a[2] == "1":
                train_set.append(((constructed_graphs, g), one))
            else:
                train_set.append((((constructed_graphs, g), zero)))
    except Exception as e:
        print(e)
        continue


with open(f'human_part_train.pkl', 'wb') as f:
    pickle.dump(train_set, f)

train_set = []
i = 1
for item in raw_data_test:
    print(i)
    i += 1
    try:
        a = item.split()
        smile = a[0]
        sequence = a[1]
        pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()[:-1]
        if pdb_code != "6g5i" and pdb_code != "5t0j" and pdb_code != "5wve":
            if previous_pdb != pdb_code:
                pdbl = PDBList()
                pdbl.retrieve_pdb_file(
                    pdb_code, pdir='./pdbs/', overwrite=True, file_format="pdb"
                )
                # Rename file to .pdb from .ent
                os.rename(
                    './pdbs/' + "pdb" + pdb_code + ".ent", './pdbs/' + pdb_code + ".pdb"
                )
                # Assert file has been downloaded
                assert any(pdb_code in s for s in os.listdir('./pdbs/'))
                # print(f"Downloaded PDB file for: {pdb_code}")
                _, _, constructed_graphs = process_protein(f"./pdbs/{pdb_code}.pdb")

                previous_pdb = pdb_code

            g = smiles_to_bigraph(smile, node_featurizer=node_featurizer)
            g = dgl.add_self_loop(g)
            if a[2] == "1":
                train_set.append(((constructed_graphs, g), one))
            else:
                train_set.append((((constructed_graphs, g), zero)))
    except Exception as e:
        print(e)
        continue

with open(f'human_part_test.pkl', 'wb') as f:
    pickle.dump(train_set, f)


train_set = []
i = 1
for item in raw_data_valid:
    print(i)
    i += 1
    try:
        a = item.split()
        smile = a[0]
        sequence = a[1]
        pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()[:-1]
        if pdb_code != "6g5i" and pdb_code != "5t0j" and pdb_code != "5wve":
            if previous_pdb != pdb_code:
                pdbl = PDBList()
                pdbl.retrieve_pdb_file(
                    pdb_code, pdir='./pdbs/', overwrite=True, file_format="pdb"
                )
                # Rename file to .pdb from .ent
                os.rename(
                    './pdbs/' + "pdb" + pdb_code + ".ent", './pdbs/' + pdb_code + ".pdb"
                )
                # Assert file has been downloaded
                assert any(pdb_code in s for s in os.listdir('./pdbs/'))
                # print(f"Downloaded PDB file for: {pdb_code}")
                _, _, constructed_graphs = process_protein(f"./pdbs/{pdb_code}.pdb")

                previous_pdb = pdb_code

            g = smiles_to_bigraph(smile, node_featurizer=node_featurizer)
            g = dgl.add_self_loop(g)
            if a[2] == "1":
                train_set.append(((constructed_graphs, g), one))
            else:
                train_set.append((((constructed_graphs, g), zero)))
    except Exception as e:
        print(e)
        continue

with open(f'human_part_val.pkl', 'wb') as f:
    pickle.dump(train_set, f)
# print('Num train keys: ', counter)
# print('Num actual train keys: ', len(train_keys))
#

# with open('train_dude_all_decoy.pkl', 'wb') as f:
#     pickle.dump(inactive, f)
#
