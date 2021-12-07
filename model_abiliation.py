import random

from dgl.nn.pytorch.conv import GATConv, GraphConv, TAGConv, GINConv, APPNPConv
from dgl.nn import TWIRLSConv
from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling
import torch
from torch.autograd import Variable

from layers import MultiHeadAttention
import torch.nn as nn
import torch.nn.functional as F
import time
from dgl import unbatch


class DTIModelWithoutBatching(nn.Module):

    def __init__(self):
        super(DTIModelWithoutBatching, self).__init__()

        self.protein_graph_conv = nn.ModuleList()
        for i in range(14):
            self.protein_conv = TWIRLSConv(31, 31, 8, prop_step=7,  alp=1, lam=1, attention=True, num_mlp_before=2, num_mlp_after=0, dropout=0.3)

        self.ligand_graph_conv = nn.ModuleList()

        self.ligand_graph_conv.append(TWIRLSConv(74, 31, 8, prop_step=7, alp=1, lam=1, attention=True, num_mlp_before=2, num_mlp_after=0, dropout=0.3))
        self.ligand_graph_conv.append(TWIRLSConv(31, 31, 8, prop_step=7, alp=1, lam=1, attention=True, num_mlp_before=2, num_mlp_after=0, dropout=0.3))
        self.ligand_graph_conv.append(
            TWIRLSConv(31, 31, 8, prop_step=7, alp=1, lam=1, attention=True, num_mlp_before=2, num_mlp_after=0,
                       dropout=0.3))
        self.ligand_graph_conv.append(
            TWIRLSConv(31, 31, 8, prop_step=7, alp=1, lam=1, attention=True, num_mlp_before=2, num_mlp_after=0,
                       dropout=0.3))
        self.ligand_graph_conv.append(
            TWIRLSConv(31, 31, 8, prop_step=7, alp=1, lam=1, attention=True, num_mlp_before=2, num_mlp_after=0,
                       dropout=0.3))
        self.ligand_graph_conv.append(
            TWIRLSConv(31, 31, 8, prop_step=7, alp=1, lam=1, attention=True, num_mlp_before=2, num_mlp_after=0,
                       dropout=0.3))
        self.ligand_graph_conv.append(
            TWIRLSConv(31, 31, 8, prop_step=7, alp=1, lam=1, attention=True, num_mlp_before=2, num_mlp_after=0,
                       dropout=0.3))
        self.ligand_graph_conv.append(
            TWIRLSConv(31, 31, 8, prop_step=7, alp=1, lam=1, attention=True, num_mlp_before=2, num_mlp_after=0,
                       dropout=0.3))
        self.pooling_ligand = nn.Linear(31, 1)
        self.pooling_protein = nn.Linear(31, 1)

        self.dropout = 0.3

        self.bilstm = nn.LSTM(31, 31, num_layers=2, bidirectional=True, dropout=0.3)

        self.fc_in = nn.Linear(8680, 4340) #1922

        self.fc_out = nn.Linear(4340, 1)
        self.attention = MultiHeadAttention(62, 62, 2)
    #    self.W_s1 = nn.Linear(60, 45) #62
    #    self.W_s2 = nn.Linear(45, 30)

    #def attention_net(self, lstm_output):
    #    attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
    #    attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
    #    attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

    #    return attn_weight_matrix

    def forward(self, g):
        feature_protein = g[0].ndata['h']
        feature_smile = g[1].ndata['h']

        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g[0], feature_protein))

        for module in self.ligand_graph_conv:
            feature_smile = F.relu(module(g[1], feature_smile))

        pool_ligand = GlobalAttentionPooling(self.pooling_ligand)
        pool_protein = GlobalAttentionPooling(self.pooling_protein)
        protein_rep = pool_protein(g[0], feature_protein).view(-1, 31)
        ligand_rep = pool_ligand(g[1], feature_smile).view(-1, 31)
        #sequence = []
        #for item in protein_rep:
        #    sequence.append(item.view(1, 31))
        #    sequence.append(ligand_rep)

        sequence = torch.cat((ligand_rep, protein_rep), dim=0).view(1, -1, 31)
        mask = torch.eye(140, dtype=torch.uint8).view(1, 140, 140).cuda()
        mask[0, sequence.size()[1]:140, :] = 0
        mask[0, :, sequence.size()[1]:140] = 0
        mask[0, :, sequence.size()[1] - 1] = 1
        mask[0, sequence.size()[1] - 1, :] = 1
        mask[0,  sequence.size()[1] - 1,  sequence.size()[1] - 1] = 0
        sequence = F.pad(input=sequence, pad=(0, 0, 0, 140 - sequence.size()[1]), mode='constant', value=0)
        sequence = sequence.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(4, 1, 31).cuda())
        c_0 = Variable(torch.zeros(4, 1, 31).cuda())

        output, _ = self.bilstm(sequence, (h_0, c_0))

        output = output.permute(1, 0,  2)

        out = self.attention(output, mask=mask)
        #attn_weight_matrix = self.attention_net(output)
        #out = torch.bmm(attn_weight_matrix, output)
        out = F.relu(self.fc_in(out.view(-1, out.size()[1]*out.size()[2])))

        out = torch.sigmoid(self.fc_out(out))
        return out


class DTITAG(nn.Module):

    def __init__(self):
        super(DTITAG, self).__init__()

        self.protein_graph_conv = nn.ModuleList()
        for i in range(5):
            self.protein_graph_conv.append(TAGConv(31, 31, 2))

        self.ligand_graph_conv = nn.ModuleList()

        self.ligand_graph_conv.append(TAGConv(74, 70, 2))
        self.ligand_graph_conv.append(TAGConv(70, 65, 2))
        self.ligand_graph_conv.append(TAGConv(65, 60, 2))
        self.ligand_graph_conv.append(TAGConv(60, 55, 2))
        self.ligand_graph_conv.append(TAGConv(55, 31, 2))
        #self.ligand_graph_conv.append(TAGConv(50, 31, 2))
        # self.ligand_graph_conv.append(TAGConv(45, 40, 2))
        # self.ligand_graph_conv.append(TAGConv(40, 31, 2))
        #self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        #self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        #self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        #self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))

        self.pooling_ligand = nn.Linear(31, 1)
        self.pooling_protein = nn.Linear(31, 1)

        self.dropout = 0.2

        self.bilstm = nn.LSTM(31, 31, num_layers=1, bidirectional=True, dropout=self.dropout)

        self.fc_in = nn.Linear(8680, 4340) #1922

        self.fc_out = nn.Linear(4340, 1)
        self.attention = MultiHeadAttention(62, 62, 2)
    #    self.W_s1 = nn.Linear(60, 45) #62
    #    self.W_s2 = nn.Linear(45, 30)

    #def attention_net(self, lstm_output):
    #    attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
    #    attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
    #    attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

    #    return attn_weight_matrix

    def forward(self, g):
        feature_protein = g[0].ndata['h']
        feature_smile = g[1].ndata['h']

        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g[0], feature_protein))

        for module in self.ligand_graph_conv:
            feature_smile = F.relu(module(g[1], feature_smile))

        pool_ligand = GlobalAttentionPooling(self.pooling_ligand)
        pool_protein = GlobalAttentionPooling(self.pooling_protein)
        protein_rep = pool_protein(g[0], feature_protein).view(-1, 31)
        ligand_rep = pool_ligand(g[1], feature_smile).view(-1, 31)
        #sequence = []
        #for item in protein_rep:
        #    sequence.append(item.view(1, 31))
        #    sequence.append(ligand_rep)

        sequence = torch.cat((ligand_rep, protein_rep), dim=0).view(1, -1, 31)
        mask = torch.eye(140, dtype=torch.uint8).view(1, 140, 140).cuda()
        mask[0, sequence.size()[1]:140, :] = 0
        mask[0, :, sequence.size()[1]:140] = 0
        mask[0, :, sequence.size()[1] - 1] = 1
        mask[0, sequence.size()[1] - 1, :] = 1
        mask[0,  sequence.size()[1] - 1,  sequence.size()[1] - 1] = 0
        sequence = F.pad(input=sequence, pad=(0, 0, 0, 140 - sequence.size()[1]), mode='constant', value=0)
        sequence = sequence.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(2, 1, 31).cuda())
        c_0 = Variable(torch.zeros(2, 1, 31).cuda())

        output, _ = self.bilstm(sequence, (h_0, c_0))

        output = output.permute(1, 0,  2)

        out = self.attention(output, mask=mask)
        #attn_weight_matrix = self.attention_net(output)
        #out = torch.bmm(attn_weight_matrix, output)
        out = F.relu(self.fc_in(out.view(-1, out.size()[1]*out.size()[2])))

        out = torch.sigmoid(self.fc_out(out))
        return out
