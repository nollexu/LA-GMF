import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import unfoldNd
from torch_geometric.nn import global_mean_pool as gap, GCNConv, AGNNConv


class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_q

        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        att = torch.bmm(dist, v)
        return att


class Attention_Layer(nn.Module):
    def __init__(self, ):
        super(Attention_Layer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, w, bias):
        # [3, 150, 256]
        # [450, 256]
        out = x.contiguous().view(x.size(0) * x.size(1), x.size(2))
        # print('attention out', out.shape)
        # [450, 2]
        out_f = F.linear(out, w, bias)
        # [450, 1]-->[3, 150]
        alpha_01 = self.attention(out).view(x.size(0), x.size(1))
        # [3, 150, 1]
        alpha = torch.unsqueeze(alpha_01, dim=2)
        # [3, 150,256]
        out = alpha.expand_as(x) * x

        return out, out_f, alpha_01


class AGNN(nn.Module):
    def __init__(self):
        super(AGNN, self).__init__()
        self.lin1 = torch.nn.Linear(512, 256)
        # self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        # x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        return x


class LA_GMF(nn.Module):
    def __init__(self):
        super(LA_GMF, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # stage 1
        self.conv_2x2x2_1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=2, stride=2),
            nn.BatchNorm3d(16))
        self.conv_3x3x3_1 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16))
        self.conv_3x3x3_2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16))

        # stage 2
        self.pool_2x2x2_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_3x3x3_3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32))
        self.conv_3x3x3_4 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32))

        # stage 3
        self.pool_2x2x2_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_3x3x3_5 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64))
        self.conv_3x3x3_6 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64))

        # stage 4
        self.pool_2x2x2_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_3x3x3_7 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1),
            nn.BatchNorm3d(128))
        self.conv_3x3x3_8 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1),
            nn.BatchNorm3d(128))

        # stage 5
        self.pool_2x2x2_5 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_3x3x3_9 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1),
            nn.BatchNorm3d(256))
        self.conv_3x3x3_10 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=1),
            nn.BatchNorm3d(256))

        self.attention_layer = Attention_Layer()

        # linear layer
        self.linear_1 = nn.Linear(8, 2)

        # Self-attention mechanism
        self.self_attention = SelfAttention(dim_q=256, dim_k=256, dim_v=256)

        # Classification layer of backbone network
        self.fc1 = nn.Linear(256, 2)

        # Classification layer of graph branches
        self.fc2 = nn.Linear(256, 2)

        # GCN corresponding to scale one
        self.gcn_1 = GCNConv(256, 256)
        # GCN corresponding to scale two
        self.gcn_2 = GCNConv(256, 256)

        # AGNN corresponding to scale one
        self.agnn1 = AGNN()
        # AGNN corresponding to scale two
        self.agnn2 = AGNN()

    # x：160*192*160
    def forward(self, x):
        # stage 1
        out = self.relu(self.conv_2x2x2_1(x))
        out_1 = self.relu(self.conv_3x3x3_1(out))
        out_2 = self.conv_3x3x3_2(out_1)
        temp_out = self.relu(out_1 + out_2)
        out_3 = torch.concat([out, temp_out], dim=1)

        # stage 2
        out_4 = self.pool_2x2x2_2(out_3)
        out_5 = self.relu(self.conv_3x3x3_3(out_4))
        out_6 = self.conv_3x3x3_4(out_5)
        temp_out = self.relu(out_5 + out_6)
        out_7 = torch.concat([out_4, temp_out], dim=1)

        # stage 3
        out_8 = self.pool_2x2x2_3(out_7)
        out_9 = self.relu(self.conv_3x3x3_5(out_8))
        out_10 = self.conv_3x3x3_6(out_9)
        temp_out = self.relu(out_9 + out_10)
        out_11 = torch.concat([out_8, temp_out], dim=1)

        out_12 = self.pool_2x2x2_4(out_11)
        out_13 = self.relu(self.conv_3x3x3_7(out_12))
        out_14 = self.conv_3x3x3_8(out_13)
        temp_out = self.relu(out_13 + out_14)
        out_15 = torch.concat([out_12, temp_out], dim=1)

        out_16 = self.pool_2x2x2_5(out_15)
        out_17 = self.relu(self.conv_3x3x3_9(out_16))
        out_18 = self.relu(self.conv_3x3x3_10(out_17))

        # 3*256*150
        out_18_new = out_18.view(out_18.size(0), out_18.size(1), -1)
        # 3*150*256
        out_18_new = out_18_new.transpose(2, 1)
        out, f, alpha = self.attention_layer(out_18_new, self.fc1.weight, self.fc1.bias)
        # 3*256
        out = out.transpose(2, 1).mean(2)
        # 3*256
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        # 3*128*10*12*10-->3*(150*128)*8
        unfold1 = unfoldNd.UnfoldNd(kernel_size=2, dilation=1, padding=0, stride=2)
        out_14 = torch.permute(unfold1(out_14), (0, 2, 1)).reshape(out_14.shape[0], 128 * 150, 8)
        # 3*(150*128)*8-->3*(150*128)*2-->3*150*256
        out_14 = self.linear_1(out_14).reshape(out_14.shape[0], 150, -1)

        # flatten attention
        flatten_attention = alpha.reshape(x.shape[0], -1)
        # construct graph
        final_edge_index_1 = 0
        final_edge_index_2 = 0
        final_data_1 = 0
        final_data_2 = 0
        final_batch_1 = 0
        final_batch_2 = 0
        # The weight of the edge is not used. You can try using it.
        final_edge_weight_1 = 0
        final_edge_weight_2 = 0

        for index in range(x.shape[0]):
            # 256*150
            patches_1 = out_18[index].reshape((256, -1))
            # 150*256
            patches_1 = torch.permute(patches_1, dims=(1, 0))
            # 150*256
            patches_2 = out_14[index]

            attention_value, attention_index = torch.sort(flatten_attention[index], dim=-1, descending=True)
            # k=10
            idx = attention_index[:10]
            # topk
            patches_1 = patches_1[idx]
            patches_2 = patches_2[idx]
            # Using cosine similarity to create a graph
            adj_matrix_1 = torch.cosine_similarity(patches_1.unsqueeze(1), patches_1.unsqueeze(0), dim=-1,
                                                   eps=1e-08)
            adj_matrix_2 = torch.cosine_similarity(patches_2.unsqueeze(1), patches_2.unsqueeze(0), dim=-1,
                                                   eps=1e-08)
            # filter edge weights
            adj_matrix_1 = torch.where(adj_matrix_1 >= 0.4, adj_matrix_1, 0)
            adj_matrix_2 = torch.where(adj_matrix_2 >= 0, adj_matrix_2, 0)

            # Convert dense adjacency matrix to sparse representation
            edge_index_1, edge_weight_1 = torch_geometric.utils.dense_to_sparse(adj_matrix_1)
            edge_index_2, edge_weight_2 = torch_geometric.utils.dense_to_sparse(adj_matrix_2)
            edge_index_1 = edge_index_1.cuda()
            edge_index_2 = edge_index_2.cuda()
            edge_weight_1 = edge_weight_1.cuda()
            edge_weight_2 = edge_weight_2.cuda()
            if isinstance(final_edge_index_1, int):
                final_edge_index_1 = edge_index_1
                final_edge_index_2 = edge_index_2
                final_edge_weight_1 = edge_weight_1
                final_edge_weight_2 = edge_weight_2
                final_data_1 = patches_1
                final_data_2 = patches_2
                final_batch_1 = edge_index_1.new_zeros(patches_1.shape[0])
                final_batch_2 = edge_index_2.new_zeros(patches_2.shape[0])
            else:
                # suppose the previous graph had 5 nodes:
                # [0, 0, 0]-->[5,5,5]
                # [1, 2, 3]-->[6,7,8]
                edge_index_1 = edge_index_1 + final_data_1.shape[0]
                edge_index_2 = edge_index_2 + final_data_2.shape[0]
                # concat edge_index
                final_edge_index_1 = torch.cat((final_edge_index_1, edge_index_1), dim=1)
                final_edge_index_2 = torch.cat((final_edge_index_2, edge_index_2), dim=1)
                final_edge_weight_1 = torch.cat((final_edge_weight_1, edge_weight_1))
                final_edge_weight_2 = torch.cat((final_edge_weight_2, edge_weight_2))

                # concat data
                final_data_1 = torch.cat((final_data_1, patches_1), dim=0)
                final_data_2 = torch.cat((final_data_2, patches_2), dim=0)
                batch_1 = edge_index_1.new_zeros(patches_1.shape[0]) + index
                batch_2 = edge_index_2.new_zeros(patches_2.shape[0]) + index

                # concat batch
                final_batch_1 = torch.cat((final_batch_1, batch_1))
                final_batch_2 = torch.cat((final_batch_2, batch_2))
        # dropout
        final_data_1 = F.dropout(final_data_1, p=0.5, training=self.training)
        final_data_2 = F.dropout(final_data_2, p=0.5, training=self.training)
        # GCN
        graph_out_1 = F.relu(self.gcn_1(final_data_1, final_edge_index_1))
        graph_out_2 = F.relu(self.gcn_2(final_data_2, final_edge_index_2))

        # self_attention
        graph_out_1 = torch.unsqueeze(graph_out_1, dim=1)
        graph_out_2 = torch.unsqueeze(graph_out_2, dim=1)
        fusion_graph_out = torch.concat([graph_out_1, graph_out_2], dim=1)
        weighted_graph_out = self.self_attention(x=fusion_graph_out)
        graph_out_1 = weighted_graph_out[:, 0, :]
        graph_out_2 = weighted_graph_out[:, 1, :]

        # AGNN
        graph_out = torch.concat([graph_out_1, graph_out_2], dim=-1)
        graph_out1 = self.agnn1(graph_out, final_edge_index_1)
        graph_out2 = self.agnn2(graph_out, final_edge_index_2)
        graph_out1 = gap(graph_out1, batch=final_batch_1)
        graph_out2 = gap(graph_out2, batch=final_batch_1)
        graph_out = graph_out1 + graph_out2

        graph_final_out = self.fc2(graph_out.view(x.shape[0], 256))

        return out, f, alpha, graph_final_out
