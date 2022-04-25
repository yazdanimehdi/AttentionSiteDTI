import os
import pickle
from collections import OrderedDict
import random
import glob
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
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7]) +
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


valid_keys = glob.glob('./all/*')
print(valid_keys)
valid_keys = [v.split('/')[-1] for v in valid_keys]

dude_gene = list(OrderedDict.fromkeys([v.split('_')[0] for v in valid_keys]))
test_fold = "tryb1_2zebA_full mcr_2oaxE_full bace1_3h0bA_full cxcr4_3oduA_full thb_1q4xA_full andr_2hvcA_full rxra_3ozjA_full esr2_2fszA_full mmp13_2pjtC_full pparg_2i4zA_full esr1_3dt3B_full prgr_3kbaA_full hivpr_1mtbA_full reni_3g6zB_full fa10_2p16A_full dpp4_2i78A_full adrb2_3ny8A_full fa7_1wqvH_full ppara_2p54A_full thrb_1ypeH_full ada17_2fv5A_full ace_3bklA_full urok_1sqtA_full gcr_3bqdA_full drd3_3pblA_full aa2ar_3emlA_full lkha4_3ftxA_full try1_2zq1A_full ppard_2znpA_full cp2c9_1r9oA_full cp3a4_1w0fA_full casp3_1rhrB_full adrb1_2vt4A_full "
test_list = test_fold.split()
test_dude_gene = []
for item in test_list:
    test_dude_gene.append(item.split("_")[0])
print(test_dude_gene)
train_dude_gene = [p for p in dude_gene if p not in test_dude_gene]
print(len(train_dude_gene))
train_keys = [k for k in valid_keys if k.split('_')[0] in train_dude_gene]
test_keys = [k for k in valid_keys if k.split('_')[0] in test_dude_gene]

node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')

zero = np.eye(2)[1]
one = np.eye(2)[0]

actives = []
inactive = []
counter = 0

with open("receptor_pdb_dict.pkl", 'rb') as fp:
    receptor_dict = pickle.load(fp)

for key in train_keys:
    try:
        try:
            _, _, constructed_graphs = process_protein(f"./all/{key}/receptor.pdb")
        except:
            pdb_code = receptor_dict[key.upper()]
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
            print(f"Downloaded PDB file for: {pdb_code}")
            _, _, constructed_graphs = process_protein(f"./pdbs/{pdb_code}.pdb")
        with open(f"./all/{key}/actives_final.ism", 'r') as fr:
            actives_string = fr.read()
        actives_string_list = actives_string.split("\n")

        with open(f"./all/{key}/decoys_final.ism", 'r') as fr:
            decoys_string = fr.read()

        decoys_string_list = decoys_string.split("\n")

    except Exception as e:
        print(e)
        counter += 1
        continue

    for smile in actives_string_list:
        try:
            g = smiles_to_bigraph(smile.split()[0], node_featurizer=node_featurizer)
            g = dgl.add_self_loop(g)
            actives.append(((constructed_graphs, g), one))
        except:
            pass

    for smile in random.sample(decoys_string_list, len(actives_string_list)):
        try:
            g = smiles_to_bigraph(smile.split()[0], node_featurizer=node_featurizer)
            g = dgl.add_self_loop(g)
            inactive.append(((constructed_graphs, g), zero))
        except:
            pass

print('Num Invalid: ', counter)
print('Num actual train keys: ', len(train_keys))

with open('train_new_dude_balanced_all2_active.pkl', 'wb') as f:
    pickle.dump(actives, f)
with open('train_new_dude_balanced_all2_decoy.pkl', 'wb') as f:
    pickle.dump(inactive, f)

actives = []
inactive = []
for key in test_keys:
    try:
        try:
            _, _, constructed_graphs = process_protein(f"./all/{key}/receptor.pdb")
        except:
            pdb_code = receptor_dict[key.upper()]
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
            print(f"Downloaded PDB file for: {pdb_code}")
            _, _, constructed_graphs = process_protein(f"./pdbs/{pdb_code}.pdb")
        with open(f"./all/{key}/actives_final.ism", 'r') as fr:
            actives_string = fr.read()
        actives_string_list = actives_string.split("\n")

        with open(f"./all/{key}/decoys_final.ism", 'r') as fr:
            decoys_string = fr.read()

        decoys_string_list = decoys_string.split("\n")

    except Exception as e:
        print(e)
        counter += 1
        continue

    for smile in actives_string_list:
        try:
            g = smiles_to_bigraph(smile.split()[0], node_featurizer=node_featurizer)
            g = dgl.add_self_loop(g)
            actives.append(((constructed_graphs, g), one))
        except:
            pass

    random_selector = np.random.randint(len(decoys_string_list) - len(actives_string_list))

    for smile in decoys_string_list:
        try:
            g = smiles_to_bigraph(smile.split()[0], node_featurizer=node_featurizer)
            g = dgl.add_self_loop(g)
            inactive.append(((constructed_graphs, g), zero))
        except:
            pass

with open('test_new_dude_all_active_none_pdb.pkl', 'wb') as f:
    pickle.dump(actives, f)
with open('test_new_dude_all_decoy_none_pdb.pkl', 'wb') as f:
    pickle.dump(inactive, f)
