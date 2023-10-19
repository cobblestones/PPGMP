from torch.nn import Module
from torch import nn
import torch
import math
import sys
sys.path.append('..')

from model import GCN
import utils.util as util
import numpy as np


class AttModel(Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcn_spatial_1 = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)
        self.gcn_temporal_1 = GCN.GCN(input_feature=in_features, hidden_feature=d_model, p_dropout=0.3,
                            num_stage=num_stage,
                           node_n=(dct_n) * 2)


        self.gcn_spatial_2 = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                                     num_stage=num_stage,
                                     node_n=in_features)
        self.gcn_temporal_2 = GCN.GCN(input_feature=in_features, hidden_feature=d_model, p_dropout=0.3,
                                      num_stage=num_stage,
                                      node_n=(dct_n) * 2)




    def forward(self, src, output_n=25, input_n=50, itera=1):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        outputs_gcn_0 = []
        outputs_gcn_1 = []
        outputs_gcn_2 = []

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        for i in range(itera):

            dct_n = self.dct_n
            src = src[:, :input_n]  # [bs,in_n,dim]
            src_tmp = src.clone()
            bs = src.shape[0]
            src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
            src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()
            vn = input_n - self.kernel_size - output_n + 1
            vl = self.kernel_size + output_n
            idx = np.expand_dims(np.arange(vl), axis=0) + \
                  np.expand_dims(np.arange(vn), axis=1)
            src_value_tmp = src_tmp[:, idx].clone().reshape(
                [bs * vn, vl, -1])
            src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
                [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
                [bs, vn, -1])  # [32,40,66*11]

            idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
            key_tmp = self.convK(src_key_tmp / 1000.0)
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            # att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            dct_att_tmp = torch.matmul(score_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])

            out_gcn_0 = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                     dct_att_tmp[:, :, :dct_n].transpose(1, 2))

            input_gcn = src_tmp[:, idx]
            dct_initial_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

            dct_in_tmp = torch.cat([dct_initial_tmp, dct_att_tmp], dim=-1)
            spatial_gcn_out_1 = self.gcn_spatial_1(dct_in_tmp)
            temporal_gcn_out_1 = self.gcn_temporal_1(spatial_gcn_out_1.permute(0,2,1))
            out_gcn_1 = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0), temporal_gcn_out_1.permute(0,2,1)[:, :, :dct_n].transpose(1, 2))

            dct_out_gcn_1 = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), out_gcn_1).transpose(1, 2)
            dct_in_tmp = torch.cat([dct_initial_tmp, dct_out_gcn_1], dim=-1)
            spatial_gcn_out_2 = self.gcn_spatial_2(dct_in_tmp)
            temporal_gcn_out_2 = self.gcn_temporal_2(spatial_gcn_out_2.permute(0, 2, 1))
            out_gcn_2 = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                     temporal_gcn_out_2.permute(0, 2, 1)[:, :, :dct_n].transpose(1, 2))



            outputs_gcn_0.append(out_gcn_0.unsqueeze(2))
            outputs_gcn_1.append(out_gcn_1.unsqueeze(2))
            outputs_gcn_2.append(out_gcn_2.unsqueeze(2))

            if itera > 1:
                out_tmp = out_gcn_2.clone()[:, 0 - output_n:]
                src = torch.cat([src, out_tmp], dim=1)
                src = src[:, -input_n:]

        outputs_gcn_0 = torch.cat(outputs_gcn_0, dim=2)
        outputs_gcn_1 = torch.cat(outputs_gcn_1, dim=2)
        outputs_gcn_2 = torch.cat(outputs_gcn_2, dim=2)


        return outputs_gcn_0,outputs_gcn_1,outputs_gcn_2,score_tmp

if __name__=="__main__":
    model=AttModel(in_features=66,kernel_size=10,d_model=256,num_stage=2,dct_n=20).cuda()
    x=torch.rand([128,60,66]).cuda()
    y0,y1,y2,weight=model(x,input_n=50, output_n=10, itera=3)
    print('y1.shape:{}'.format(y1.shape))
    print('weight.shape:{}'.format(weight.shape))