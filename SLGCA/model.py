import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.to_dense()
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)

class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu, tau = 0.5):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.tau = tau

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        # self.weight3 = Parameter(torch.FloatTensor(self.out_features, self.out_features))
        self.reset_parameters()
        self.sigm = nn.Sigmoid()
        self.dc = InnerProductDecoder(0.1, self.sigm)
        self.read = AvgReadout()

        ##############################################################
        ##############################################################
        self.decoder = nn.Sequential(
            # nn.BatchNorm1d(self.out_features),
            nn.Dropout(0.3),
            nn.Linear(self.out_features, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features)
        )

        # self.decoder = torch.nn.Linear(self.out_features, self.out_features)

        ##############################################################
        ##############################################################

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)

        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)

        #############################################

        #############################################


        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)

        emb = self.act(z)
        emb_a = self.act(z_a)

        dec = self.decoder(emb)
        dec_a = self.decoder(emb_a)

        g = self.read(emb,self.graph_neigh)
        ret = self.sigm(g)
        g_a = self.read(emb_a, self.graph_neigh)
        ret_a = self.sigm(g_a)

        A_rec = self.dc(emb)


        return hiden_emb, h, dec, dec_a, A_rec,ret,ret_a

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, h1: torch.Tensor, h2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def nei_con_loss(self, z1: torch.Tensor, z2: torch.Tensor, adj):
        '''neighbor contrastive loss'''
        adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
        # Replace the thresholding with probability >= 0.5
        adj = (adj >= 0.3).float()  # Convert to binary: 1 if probability >= 0.5, 0 otherwise
        nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
        nei_count = torch.squeeze(torch.tensor(nei_count))

        f = lambda x: torch.exp(x / self.tau)
        intra_view_sim = f(self.sim(z1, z1))
        inter_view_sim = f(self.sim(z1, z2))

        loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
                intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
        loss = loss / nei_count  # divided by the number of positive pairs for each node

        return -torch.log(loss)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor, adj,
                         mean: bool = True):
        l1 = self.nei_con_loss(z1, z2, adj)
        l2 = self.nei_con_loss(z2, z1, adj)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def nei_con_loss_bias(self, z1: torch.Tensor, z2: torch.Tensor, adj, pseudo_labels):
        '''neighbor contrastive loss'''
        adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
        adj[adj > 0.5] = 1
        nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
        nei_count = torch.squeeze(torch.tensor(nei_count))

        f = lambda x: torch.exp(x / self.tau)
        intra_view_sim = f(self.sim(z1, z1))
        inter_view_sim = f(self.sim(z1, z2))

        # Create a mask for negative samples with different pseudo labels
        negative_mask = (pseudo_labels.view(-1, 1) != pseudo_labels.view(1, -1)).float()

        # Apply the mask to intra_view_sim and inter_view_sim
        masked_intra_view_sim = intra_view_sim * negative_mask
        masked_inter_view_sim = inter_view_sim * negative_mask

        loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
                masked_intra_view_sim.sum(1) + masked_inter_view_sim.sum(1) - intra_view_sim.diag())
        loss = loss / nei_count  # divided by the number of positive pairs for each node

        return -torch.log(loss)

    def contrastive_loss_bias(self, z1: torch.Tensor, z2: torch.Tensor, adj, pseudo_labels,
                              mean: bool = True):
        l1 = self.nei_con_loss_bias(z1, z2, adj, pseudo_labels)
        l2 = self.nei_con_loss_bias(z2, z1, adj, pseudo_labels)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


class Encoder_sparse(Module):
    """
    Sparse version of Encoder
    """

    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.sigm = nn.Sigmoid()

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        # self.weight3 = Parameter(torch.FloatTensor(self.out_features, self.out_features))
        self.reset_parameters()
        self.read = AvgReadout()


        ##############################################################
        ##############################################################
        self.decoder = nn.Sequential(
            # nn.BatchNorm1d(self.out_features),
            nn.Dropout(0.3),
            nn.Linear(self.out_features, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features)
        )


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)

        hiden_emb = z
        emb = self.act(z)


        h = torch.mm(emb, self.weight2)
        h = torch.spmm(adj, h)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)

        emb_a = self.act(z_a)


        dec = self.decoder(emb)
        dec_a = self.decoder(emb_a)

        g = self.read(emb, self.graph_neigh)
        ret = self.sigm(g)
        g_a = self.read(emb_a, self.graph_neigh)
        ret_a = self.sigm(g_a)

        return hiden_emb, h, dec, dec_a,ret,ret_a


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def contrastive_loss(self, h1: torch.Tensor, h2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret