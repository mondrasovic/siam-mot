import torch

from torch import nn
from torch.nn import functional as F


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
        select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def log_softmax(cls_logits):
    b, a2, h, w = cls_logits.size()
    cls_logits = cls_logits.view(b, 2, a2 // 2, h, w)
    cls_logits = cls_logits.permute(0, 2, 3, 4, 1).contiguous()
    cls_logits = F.log_softmax(cls_logits, dim=4)
    return cls_logits


def features_to_emb(features: torch.Tensor) -> torch.Tensor:
    """Computes embedding vectors from tracker template (exemplar) features.
    For each feature tensor in a batch, it applies global average pooling along
    the channel dimension. Afterwards, it L2-normalizes the vectors to project
    them onto a unit hypersphere.

    Args:
        features (torch.Tensor): Template features of shape [B, C, S, S].

    Returns:
        torch.Tensor: Embedding vectors of shape [B, C].
    """
    batch_size, n_channels, kernel_size, _ = features.shape
    avg = F.avg_pool2d(features, kernel_size=kernel_size)   # [B, C, 1, 1]
    avg  = avg.reshape((batch_size, n_channels))  # [B, C]
    norm = torch.linalg.norm(avg, dim=1)  # [B,]
    emb = avg / norm[..., None]  # [B, C]
    
    return emb


def _pairwise_l2_dist(
    embs: torch.Tensor,
    *,
    squared: bool = False,
    eps: float = 1e-16
) -> torch.Tensor:
    """Computes a 2D matrix of L2 or squared L2 distances between all the
    embeddings.

    Args:
        embs (torch.Tensor): embeddings of shape [B,E]
        squared (bool, optional): If True, the output is the pairwise squared L2
        distance, if False, then standard L2 distance is computed. Defaults to
        False.
        eps (float, optional): Small value to add to the zero distances to
        prevent infinite gradients after applying sqrt(). Defaults to 1e-16.

    Returns:
        torch.Tensor: A 2D distance matrix of shape [B,B].
    """
    dot_product = torch.matmul(embs, embs.T)  # [B,B]
    square_norm = torch.diag(dot_product)  # [B,]

    # Apply the l2 norm formula using the dot product:
    # ||A - B||^2 = ||A||^2 - 2<A,B> + ||B||^2
    distances = (
        torch.unsqueeze(square_norm, dim=1) -  # [B,1]
        (2 * dot_product) +  # [B,B]
        torch.unsqueeze(square_norm, dim=0)  # [1,B]
    )

    # Due to potential errors caused by numerical instability, some values may
    # have become negative. Thus, we have to make sure the min. value is zero.
    zero = torch.tensor(0.0)
    distances = torch.maximum(distances, zero)  # [B,B]

    if not squared:
        # Since the gradient of sqrt(0) is infinite, we, therefore, have to
        # add a small epsilon to the zero terms to prevent this.
        zeroes_mask = ((distances - zero) < eps).float()  # [B,B]
        distances += zeroes_mask * eps
        
        distances = torch.sqrt(distances)  # [B,B]

        # Set all the "zero" values back to zero after adding the epsilon value.
        distances *= (1.0 - zeroes_mask)
    
    return distances


def _get_anchor_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    """Generates a 2D mask where M[a,p] is True iff anchor (a) and positive (p)
    have identical labels but distinct indices, i.e., belong to different
    objects.

    Args:
        labels (torch.Tensor): Labels of shape [B,].

    Returns:
        torch.Tensor: 2D boolean mask of shape[B,B].
    """
    labels_eq_mask = (labels[..., None] == labels[None, ...])  # [B,B]
    idxs_neq_mask = ~torch.eye(
        len(labels), dtype=torch.bool, device=labels.device
    )  # [B,B]
    anchor_positive_mask = (labels_eq_mask & idxs_neq_mask)  # [B,B]

    return anchor_positive_mask


def _get_anchor_negative_mask(labels: torch.Tensor) -> torch.Tensor:
    """Generates a 2D mask where M[a,n] is True iff anchor (a) and negative (n)
    have distinct labels.

    Args:
        labels (torch.Tensor): Labels of shape [B,].

    Returns:
        torch.Tensor: 2D boolean mask of shape[B,B].
    """
    anchor_negative_mask = (labels[..., None] != labels[None, ...])  # [B,B]

    return anchor_negative_mask


# def get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
#     idxs_neq_mask = ~torch.eye(len(labels), dtype=torch.bool)  # [B,B]
#     idx_i_neq_j_mask = torch.unsqueeze(idxs_neq_mask, dim=2)  # [B,B,1]
#     idx_i_neq_k_mask = torch.unsqueeze(idxs_neq_mask, dim=1)  # [B,1,B]
#     idx_j_neq_k_mask = torch.unsqueeze(idxs_neq_mask, dim=0)  # [1,B,B]
#     triplet_idxs_neq_mask = (
#         idx_i_neq_j_mask & idx_i_neq_k_mask & idx_j_neq_k_mask  # [B,B,B]
#     )

#     labels_eq_mask = (labels[..., None] == labels[None, ...])  # [B,B]
#     label_i_eq_j = torch.unsqueeze(labels_eq_mask, dim=2)  # [B,B,1]
#     label_i_neq_k = ~torch.unsqueeze(labels_eq_mask, dim=1)  # [B,1,B]
#     triplet_labels_valid_mask = (label_i_eq_j & label_i_neq_k)  # [B,B,B]

#     triplet_mask = (
#         triplet_idxs_neq_mask & triplet_labels_valid_mask  # [B,B,B]
#     )

#     return triplet_mask

class SemiHardTripletLoss(nn.Module):
    """Triplet loss with hard negative mining."""

    def __init__(self, margin: float = 1.0, squared: bool = True) -> None:
        """Constructor.

        Args:
            margin (float, optional): Margin of separation between positive and
            negative samples. Defaults to 1.0.
            squared (bool, optional): If True, the  the pairwise squared L2
            distance is used, if False, then standard L2 distance is computed.
            Defaults to True.
        """
        super().__init__()

        self.margin: float = margin
        self.squared: bool = squared
    
    def forward(self, embs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes the loss between all the hard triplets. For a given anchor
        sample, a hard triplet is formed by finding the hardest positive, i.e,
        a different sample with the same label which has the maximum distance;
        and also the hardest negative, .e., a different sample with a distinct
        label which has the minimum distance.

        Args:
            embs (torch.Tensor): Embeddings of shape [B,E].
            labels (torch.Tensor): Labels of shape [B,].

        Returns:
            torch.Tensor: Triplet loss scalar.
        """
        pairwise_dist = _pairwise_l2_dist(embs, squared=self.squared)

        anchor_positive_mask = _get_anchor_positive_mask(
            labels
        ).float()  # [B,B]
        anchor_positive_dist = pairwise_dist * anchor_positive_mask  # [B,B]
        hardest_positive_dist = torch.amax(
            anchor_positive_dist, dim=1, keepdim=True
        )  # [B,1]

        anchor_negative_mask = _get_anchor_negative_mask(
            labels
        ).float()  # [B,B]
        max_anchor_negative_dist = torch.amax(
            pairwise_dist, dim=1, keepdim=True
        )  # [B,1]
        anchor_negative_dist = (
            pairwise_dist +
            (1 - anchor_negative_mask) * max_anchor_negative_dist
        )  # [B,B]
        hardest_negative_dist = torch.amin(
            anchor_negative_dist, dim=1, keepdim=True
        )  # [B,1]

        triplet_loss = torch.clamp(
            hardest_positive_dist - hardest_negative_dist + self.margin, min=0
        )  # [B,1]
        triplet_loss = torch.mean(triplet_loss)  # [c]

        return triplet_loss


class BalancedMarginContrastiveLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.5,
        eps: float = 1e-8
    ) -> None:
        super().__init__()

        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = eps
    
    def forward(self, embs, ids):
        idxs = torch.arange(0, len(embs))
        idx_pairs = torch.combinations(idxs, 2)
        emb_pairs = embs[idx_pairs]

        pair_dist = torch.norm(emb_pairs[:, 0, :] - emb_pairs[:, 1, :], dim=1)

        ids_first = ids[idx_pairs[:, 0]]
        ids_second = ids[idx_pairs[:, 1]]
        neg_pairs_mask = (ids_first != ids_second)

        labels = torch.ones_like(pair_dist)
        labels[neg_pairs_mask] = -1

        n_neg = torch.sum(neg_pairs_mask).item()
        n_pos = len(neg_pairs_mask) - n_neg

        pos_weight = 1.0 / (n_pos + self.eps)
        neg_weight = 1.0 / (n_neg + self.eps)

        weights = torch.full_like(pair_dist, pos_weight)
        weights[neg_pairs_mask] = neg_weight
        weights /= weights.sum()

        loss = torch.sum(
            weights * 
            torch.clip(
                self.alpha + labels * (pair_dist - self.beta), min=0
            )
        )

        return loss


class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_l = pred[:, 0]
        pred_t = pred[:, 1]
        pred_r = pred[:, 2]
        pred_b = pred[:, 3]
        
        target_l = target[:, 0]
        target_t = target[:, 1]
        target_r = target[:, 2]
        target_b = target[:, 3]
        
        target_area = (target_l + target_r) * (target_t + target_b)
        pred_area = (pred_l + pred_r) * (pred_t + pred_b)
        
        w_intersect = torch.min(pred_l, target_l) + torch.min(pred_r, target_r)
        h_intersect = torch.min(pred_b, target_b) + torch.min(pred_t, target_t)
        
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        
        losses = -torch.log((area_intersect + 1.) / (area_union + 1.))
        
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            return losses.mean()


class EMMLossComputation(object):
    def __init__(self, cfg):
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()

        feature_emb_loss_name = cfg.MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS
        if feature_emb_loss_name == 'none':
            self.emb_loss_func = None
        elif feature_emb_loss_name == 'contrastive':
            self.emb_loss_func = BalancedMarginContrastiveLoss()
        elif feature_emb_loss_name == 'triplet':
            self.emb_loss_func = SemiHardTripletLoss()
        else:
            raise ValueError('unrecognized embedding loss function')

        self.cfg = cfg
        self.pos_ratio = cfg.MODEL.TRACK_HEAD.EMM.CLS_POS_REGION
        self.loss_weight = cfg.MODEL.TRACK_HEAD.EMM.TRACK_LOSS_WEIGHT
    
    def prepare_targets(self, points, src_bbox, gt_bbox):
        cls_labels, reg_targets = self.compute_targets(
            points, src_bbox, gt_bbox
        )
        
        return cls_labels, reg_targets
    
    def compute_targets(self, locations, src_bbox, tar_bbox):
        xs, ys = locations[:, :, 0], locations[:, :, 1]
        
        num_boxes, num_locations, _ = locations.shape
        cls_labels = torch.zeros(
            (num_boxes, num_locations),
            dtype=torch.int64, device=locations.device
        )
        
        _l = xs - tar_bbox[:, 0:1].float()
        _t = ys - tar_bbox[:, 1:2].float()
        _r = tar_bbox[:, 2:3].float() - xs
        _b = tar_bbox[:, 3:4].float() - ys
        
        s1 = _l > self.pos_ratio * (
            (tar_bbox[:, 2:3] - tar_bbox[:, 0:1]) / 2).float()
        s2 = _r > self.pos_ratio * (
            (tar_bbox[:, 2:3] - tar_bbox[:, 0:1]) / 2).float()
        s3 = _t > self.pos_ratio * (
            (tar_bbox[:, 3:4] - tar_bbox[:, 1:2]) / 2).float()
        s4 = _b > self.pos_ratio * (
            (tar_bbox[:, 3:4] - tar_bbox[:, 1:2]) / 2).float()
        
        is_in_pos_boxes = s1 * s2 * s3 * s4
        cls_labels[is_in_pos_boxes == 1] = 1
        
        reg_targets = torch.stack([_l, _t, _r, _b], dim=2)
        
        return cls_labels.contiguous(), reg_targets.contiguous()
    
    @staticmethod
    def compute_centerness_targets(reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)
    
    @staticmethod
    def normalize_regression_outputs(src_bbox, regression_outputs):
        # normalize the regression targets
        half_src_box_w = (src_bbox[:, 2:3] - src_bbox[:, 0:1]) / 2. + 1e-10
        half_src_box_h = (src_bbox[:, 3:4] - src_bbox[:, 1:2]) / 2. + 1e-10
        assert (all(half_src_box_w > 0))
        assert (all(half_src_box_h > 0))
        
        regression_outputs[:, :, 0] = (regression_outputs[:, :,
                                       0] / half_src_box_w) * 128
        regression_outputs[:, :, 1] = (regression_outputs[:, :,
                                       1] / half_src_box_h) * 128
        regression_outputs[:, :, 2] = (regression_outputs[:, :,
                                       2] / half_src_box_w) * 128
        regression_outputs[:, :, 3] = (regression_outputs[:, :,
                                       3] / half_src_box_h) * 128
        
        return regression_outputs
    
    def __call__(
        self, locations, box_cls, box_regression, centerness, src, targets,
        embs=None, ids=None
    ):
        cls_labels, reg_targets = self.prepare_targets(locations, src, targets)
        
        box_regression = (
            box_regression.permute(0, 2, 3, 1).contiguous()
        ).view(-1, 4)
        box_regression_flatten = box_regression.view(-1, 4)
        reg_targets_flatten = reg_targets.view(-1, 4)
        cls_labels_flatten = cls_labels.view(-1)
        centerness_flatten = centerness.view(-1)
        
        in_box_inds = torch.nonzero(cls_labels_flatten > 0).squeeze(1)
        box_regression_flatten = box_regression_flatten[in_box_inds]
        reg_targets_flatten = reg_targets_flatten[in_box_inds]
        centerness_flatten = centerness_flatten[in_box_inds]
        
        box_cls = log_softmax(box_cls)
        cls_loss = select_cross_entropy_loss(box_cls, cls_labels_flatten)
        
        emb_loss = torch.tensor(0., device=cls_labels.device)

        if in_box_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(
                reg_targets_flatten
            )
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )

            if self.emb_loss_func is not None:
                valid_mask = (ids >= 0)
                emb_loss = self.emb_loss_func(embs[valid_mask], ids[valid_mask])
        else:
            reg_loss = 0. * box_regression_flatten.sum()
            centerness_loss = 0. * centerness_flatten.sum()
        
        return (
            self.loss_weight * cls_loss,
            self.loss_weight * reg_loss,
            self.loss_weight * centerness_loss,
            self.loss_weight * emb_loss
        )
