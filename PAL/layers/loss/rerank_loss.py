import torch
from torch import nn
from utils.rerank import pairwise_distance
import torch.nn.functional as F
import objgraph
# def pairwise_dist(x, y):
#     # Compute pairwise distance of vectors
#     xx = (x ** 2).sum(dim=1, keepdim=True)
#     yy = (y ** 2).sum(dim=1, keepdim=True).t()
#     dist = xx + yy - 2.0 * torch.mm(x, y.t())
#     #dist = dist.clamp(min=1e-6).sqrt()  # for numerical stability
#     return dist
def intersect1d(tensor1, tensor2):
    return torch.unique(torch.cat([tensor1[tensor1 == val] for val in tensor2]))

# def pairwise_d(x, y):
#     """Compute pairwise distance between two tensors."""
#     dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(x.size(0), y.size(0)) + \
#            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(y.size(0), x.size(0)).t()
#     dist.addmm_(1, -2, x, y.t())
#     return dist


# def intersect_1d(tensor1, tensor2):
#     """Computes the intersection of two tensors along one dimension."""
#     tensor1 = tensor1.unique()
#     tensor2 = tensor2.unique()
#     intersect = torch.tensor([], dtype=torch.long).cuda()
#     for t in tensor1:
#         if torch.any(tensor2 == t):
#             intersect = torch.cat((intersect, t.unsqueeze(0)), dim=0)
#     return intersect
#
#
# # Now we'll update the re_ranking_gpu function to use this custom intersect1d function
# def re_ranking_gpu_grad_refined(q_feat, g_feat, k1=20, k2=6, lambda_value=0.3, eval_type=True):
#     # Not detaching q_feat and g_feat to ensure gradients flow back
#     feats = torch.cat([q_feat, g_feat], 0)
#     dist = pairwise_distance(feats, feats)
#
#     original_dist = dist.clone()
#     all_num = original_dist.size(0)
#     max_original_dist, _ = torch.max(original_dist, dim=0, keepdim=True)
#     original_dist = (original_dist / max_original_dist).t()
#
#     V = torch.zeros_like(original_dist)
#     query_num = q_feat.size(0)
#
#     if eval_type:
#         dist[:, query_num:] = dist.max()
#
#     initial_rank = torch.argsort(dist).cuda()
#
#     for i in range(all_num):
#         forward_k_neigh_index = initial_rank[i, :k1 + 1]
#         backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
#         fi = (backward_k_neigh_index == i).nonzero(as_tuple=True)[0]
#         k_reciprocal_index = forward_k_neigh_index[fi]
#         k_reciprocal_expansion_index = k_reciprocal_index
#         for j in k_reciprocal_index:
#             candidate = j
#             candidate_forward_k_neigh_index = initial_rank[candidate, :int(round(k1 / 2)) + 1]
#             candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
#                                                :int(round(k1 / 2)) + 1]
#             fi_candidate = (candidate_backward_k_neigh_index == candidate).nonzero(as_tuple=True)[0]
#             candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
#             if (intersect1d(candidate_k_reciprocal_index, k_reciprocal_index).size(0) >
#                     2 / 3 * candidate_k_reciprocal_index.size(0)):
#                 k_reciprocal_expansion_index = torch.unique(
#                     torch.cat((k_reciprocal_expansion_index, candidate_k_reciprocal_index), 0))
#
#         weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
#         V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)
#
#     original_dist = original_dist[:query_num, ]
#     if k2 != 1:
#         V_qe = torch.zeros_like(V)
#         for i in range(all_num):
#             V_qe[i, :] = torch.mean(V[initial_rank[i, :k2], :], dim=0)
#         V = V_qe
#         del V_qe
#
#     invIndex = []
#     for i in range(all_num):
#         invIndex.append((V[:, i] != 0).nonzero(as_tuple=True)[0])
#
#     jaccard_dist = torch.zeros_like(original_dist)
#     for i in range(query_num):
#         temp_min = torch.zeros(1, all_num).cuda()
#         indNonZero = (V[i, :] != 0).nonzero(as_tuple=True)[0]
#         indImages = [invIndex[ind] for ind in indNonZero]
#         for j, ind in enumerate(indNonZero):
#             temp = torch.min(V[i, ind], V[indImages[j], ind])
#             temp_min[0, indImages[j]] += temp
#         jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
#
#     # Computing final_dist with computation graph
#     final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
#     final_dist = final_dist[:query_num, query_num:]
#
#     # Clean up
#     del feats, dist, V, jaccard_dist, invIndex
#     torch.cuda.empty_cache()
#
#     return final_dist

#KL_dist
def KL_dist(A, B):
    """
    Compute the symmetric KL divergence between each row of matrix A and each row of matrix B using broadcasting.

    Parameters:
    - A: Tensor of shape [m, c]
    - B: Tensor of shape [n, c]

    Returns:
    - D: Tensor of shape [m, n] where D[i, j] is the symmetric KL divergence between A[i] and B[j]
    """
    # Convert rows to log probabilities and probabilities
    log_probs_A = F.log_softmax(A, dim=1)  # [m, c]
    log_probs_B = F.log_softmax(B, dim=1)  # [n, c]
    probs_A = F.softmax(A, dim=1)  # [m, c]
    probs_B = F.softmax(B, dim=1)  # [n, c]

    # Expand dimensions for broadcasting
    log_probs_A_exp = log_probs_A[:, None, :]  # [m, 1, c]
    log_probs_B_exp = log_probs_B[None, :, :]  # [1, n, c]
    probs_A_exp = probs_A[:, None, :]  # [m, 1, c]
    probs_B_exp = probs_B[None, :, :]  # [1, n, c]

    # Calculate symmetric KL divergence
    D_PQ = (log_probs_A_exp * (log_probs_A_exp.exp() - probs_B_exp)).sum(-1)  # [m, n]
    D_QP = (log_probs_B_exp * (log_probs_B_exp.exp() - probs_A_exp)).sum(-1)  # [n, m]
    D = 0.5 * (D_PQ + D_QP.transpose(0, 1))  # [m, n]

    return D







def KL1(feats):
    # 准备输入
    log_probs_P = F.log_softmax(feats, dim=1)
    normalized_P = F.softmax(feats, dim=1)

    # 初始化 KLDivLoss
    criterion = torch.nn.KLDivLoss(reduction='none')

    dist_matrix = []

    for i in range(normalized_P.shape[0]):

        kl_div_PQ = criterion(log_probs_P[i].unsqueeze(0), normalized_P).sum(dim=1) #
        kl_div_QP = criterion(log_probs_P, normalized_P[i].unsqueeze(0)).sum(dim=1) #
        skl_div = 0.5 * (kl_div_PQ + kl_div_QP)
        dist_matrix.append(skl_div)

    dist = torch.stack(dist_matrix, dim=0)
    return dist

def rerank_kl(feat1, feat2, k1=20, k2=6, lambda_value=0.3, eval_type=True):
    feats = torch.cat([feat1, feat2], 0)  #######
    dist = KL_dist(feats, feats)
    original_dist = dist.clone()
    all_num = original_dist.shape[0]

    #original_dist = original_dist / torch.max(original_dist, dim=0).values

    original_dist = original_dist.transpose(0, 1)
    V = torch.zeros_like(original_dist)  # .half()

    query_num = feat1.size(0)

    if eval_type:
        max_val = dist.max()
        dist = torch.cat((dist[:, :query_num], max_val.expand_as(dist[:, query_num:])), dim=1)
    initial_rank = torch.argsort(dist, dim=1)

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = torch.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index

        for j in k_reciprocal_index:
            candidate = j.item()
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(round(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(round(k1 / 2)) + 1]
            fi_candidate = torch.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]

            if len(intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = torch.unique(
                    torch.cat([k_reciprocal_expansion_index, candidate_k_reciprocal_index], 0))

        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = (weight / torch.sum(weight))  # .half()

    original_dist = original_dist[:query_num, ]
    # print('before')
    # objgraph.show_growth(limit=3)
    if k2 != 1:
        V_qe = torch.zeros_like(V)  # .half()
        for i in range(all_num):
            V_qe[i, :] = torch.mean(V[initial_rank[i, :k2], :], dim=0)
        V = V_qe
    invIndex = []
    for i in range(all_num):
        invIndex.append(torch.where(V[:, i] != 0)[0])
    jaccard_dist = torch.zeros_like(original_dist)  # .half()
    for i in range(query_num):
        temp_min = torch.zeros([1, all_num], device="cuda")  # .half()
        indNonZero = torch.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] += torch.min(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def rerank_dist(feat1, feat2, k1=20, k2=6, lambda_value=0.3, eval_type=True):  #q_feat, g_feat

    feats = torch.cat([feat1, feat2], 0)  #######
    dist = pairwise_distance(feats, feats)
    original_dist = dist.clone()
    all_num = original_dist.shape[0]

    #original_dist = original_dist / torch.max(original_dist, dim=0).values

    original_dist = original_dist.transpose(0, 1)
    V = torch.zeros_like(original_dist)  # .half()

    query_num = feat1.size(0)

    if eval_type:
        max_val = dist.max()
        dist = torch.cat((dist[:, :query_num], max_val.expand_as(dist[:, query_num:])), dim=1)
    initial_rank = torch.argsort(dist, dim=1)

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = torch.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index

        for j in k_reciprocal_index:
            candidate = j.item()
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(round(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(round(k1 / 2)) + 1]
            fi_candidate = torch.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]

            if len(intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = torch.unique(
                    torch.cat([k_reciprocal_expansion_index, candidate_k_reciprocal_index], 0))

        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = (weight / torch.sum(weight))  # .half()

    original_dist = original_dist[:query_num, ]
    # print('before')
    # objgraph.show_growth(limit=3)
    if k2 != 1:
        V_qe = torch.zeros_like(V)  # .half()
        for i in range(all_num):
            V_qe[i, :] = torch.mean(V[initial_rank[i, :k2], :], dim=0)
        V = V_qe
    invIndex = []
    for i in range(all_num):
        invIndex.append(torch.where(V[:, i] != 0)[0])
    jaccard_dist = torch.zeros_like(original_dist)  # .half()
    for i in range(query_num):
        temp_min = torch.zeros([1, all_num], device="cuda")  # .half()
        indNonZero = torch.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] += torch.min(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


class RerankLoss(nn.Module):
    def __init__(self, margin=0):
        super(RerankLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist_o = rerank_dist(inputs, inputs)
        dist_kl = rerank_kl(inputs, inputs)
        dist_kl = dist_kl.clamp(min=1e-12, max=10.0)
        dist = dist_o + dist_kl

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        # y = dist_an.data.new()
        # y.resize_as_(dist_an.data)
        # y.fill_(1)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        #prec = dist_an.data > dist_ap.data
        #length = torch.sqrt((inputs * inputs).sum(1)).mean()
        return loss, dist, dist_ap, dist_an
