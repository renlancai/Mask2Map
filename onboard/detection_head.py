import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32
from mmcv.cnn import Linear, bias_init_with_prob
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import multi_apply, reduce_mean, build_assigner
from mmcv.utils import TORCH_VERSION, digit_version

def is_same(images_data, good_images):
    return (torch.allclose(images_data.cpu(),  good_images.cpu(), rtol=1e-03, atol=1e-05))


def compare_two_dicts(dict1, dict2):
    for key in dict1:
        if key not in dict2:
            continue
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            return compare_two_dicts(dict1[key], dict2[key])
        if isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor):
            return print(f'{key} in debug: {is_same(dict1[key], dict2[key])}')
        else:
            continue


class CameraFeatures(nn.Module):
    def __init__(self, model):
        super(CameraFeatures, self).__init__()
        self.img_backbone = model.img_backbone
        self.img_neck = model.img_neck
        
    def forward(self, img):
        # no grid mask is needed for inference.
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        # input: tensor, output: dict
        img_feats = self.img_backbone(img)
        
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        
        #input:list, out:tuple
        img_feats = self.img_neck(img_feats)
        
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return torch.cat(img_feats_reshaped, dim=0)

from mmdet3d.ops import bev_pool
def global_bev_pooling(geom_feats, x, dx, bx, nx):
    B, N, D, H, W, C = x.shape
    Nprime = B * N * D * H * W

    # flatten x
    x = x.reshape(Nprime, C)

    # flatten indices
    geom_feats = ((geom_feats - (bx - dx / 2.0)) / dx).long()
    geom_feats = geom_feats.view(Nprime, 3)
    batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
    geom_feats = torch.cat((geom_feats, batch_ix), 1)

    # filter out points that are outside box
    kept = (
        (geom_feats[:, 0] >= 0)
        & (geom_feats[:, 0] < nx[0])
        & (geom_feats[:, 1] >= 0)
        & (geom_feats[:, 1] < nx[1])
        & (geom_feats[:, 2] >= 0)
        & (geom_feats[:, 2] < nx[2])
    )
    x = x[kept]
    geom_feats = geom_feats[kept]

    x = bev_pool(x, geom_feats, B, nx[2], nx[0], nx[1])

    # collapse Z
    final = torch.cat(x.unbind(dim=2), 1)
    
    final = final.permute(0, 1, 3, 2).contiguous()

    return final

def list2tensor(list0, device):
    import numpy as np
    list0 = np.asarray(list0)
    return torch.from_numpy(list0).unsqueeze(0).to(device)

class CameraFeaturesWithDepth(nn.Module):
    def __init__(self, model, img_metas):
        super(CameraFeaturesWithDepth, self).__init__()
        self.img_backbone = model.img_backbone
        self.img_neck = model.img_neck
        self.encoder = model.pts_bbox_head.transformer.encoder
        
        img_metas['img_shape'] = [
            (480, 800, 3), 
            (480, 800, 3),
            (480, 800, 3), 
            (480, 800, 3), 
            (480, 800, 3), 
            (480, 800, 3)]
        
        self.img_metas = [img_metas]
        
        self.dx = self.encoder.dx
        self.bx = self.encoder.bx
        self.nx = self.encoder.nx
        
    def get_img_feats(self, img):
        # no grid mask is needed for inference.
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        # input: tensor, output: dict
        img_feats = self.img_backbone(img)
        
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        
        #input:list, out:tuple
        img_feats = self.img_neck(img_feats)
        
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return torch.cat(img_feats_reshaped, dim=0)

    def get_lss_depth(self, images, \
            lidar2img, camera2ego, \
            camera_intrinsics, img_aug_matrix, lidar2ego):
        B, N, C, fH, fW = images.shape
        
        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3] # B, N, 3
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        
        img_metas = self.img_metas
        geom = self.encoder.get_geometry_v1( \
            fH, fW, rots, trans, intrins, post_rots, post_trans, lidar2ego_rots, lidar2ego_trans, img_metas)
       
        mlp_input = self.encoder.get_mlp_input( \
             camera2ego, camera_intrinsics, post_rots, post_trans)
        
        x, depth = self.encoder.get_cam_feats(images, mlp_input)
        
        return x, depth, geom


    def forward(self, img):
        device = img.device
        img_feats = self.get_img_feats(img)
        
        lidar2img = list2tensor(self.img_metas[0]['lidar2img'], device)
        camera2ego = list2tensor(self.img_metas[0]['camera2ego'], device)
        camera_intrinsics = list2tensor(self.img_metas[0]['camera_intrinsics'], device)
        img_aug_matrix = list2tensor(self.img_metas[0]['img_aug_matrix'], device)
        lidar2ego = list2tensor(self.img_metas[0]['lidar2ego'], device)
        
        # step2 :
        enhanced_feats, depth, geom = self.get_lss_depth(img_feats, lidar2img, camera2ego, \
            camera_intrinsics, img_aug_matrix, lidar2ego)
        
        return img_feats, enhanced_feats, depth, geom

class ImgBEVDownsampling(nn.Module):
    def __init__(self, model):
        super(ImgBEVDownsampling, self).__init__()
        self.downsample = model.pts_bbox_head.transformer.encoder.downsample
        
    def forward(self, img_bev_embed):
        return self.downsample(img_bev_embed)


class LidarBackboneScn(nn.Module):
    def __init__(self, model):
        super(LidarBackboneScn, self).__init__()
        self.lidar_backbone = model.lidar_modal_extractor["backbone"]
        
    def forward(self, feats, coords, batch_size, sizes):
        return self.lidar_backbone(feats, coords, batch_size, sizes)


class FeatFuser(nn.Module):
    def __init__(self, model):
        super(FeatFuser, self).__init__()
        self.fuser = model.transformer.fuser
        self.bev_h = model.bev_h
        self.bev_w = model.bev_w
    def forward(self,
            img_bev_feat,
            lidar_feat):
        if lidar_feat is None or self.fuser is None:
            return img_bev_feat
        
        # fusion
        lidar_feat = lidar_feat.permute(0, 1, 3, 2).contiguous()  # B C H W
        lidar_feat = nn.functional.interpolate(lidar_feat, size=(self.bev_h, self.bev_w), mode="bicubic", align_corners=False)
        fused_bev = self.fuser([img_bev_feat, lidar_feat])
        fused_bev = fused_bev.flatten(2).permute(0, 2, 1).contiguous()
        return fused_bev

class ImgBEVPooling(nn.Module):
    def __init__(self, model, img_metas):
        super(ImgBEVPooling, self).__init__()
        self.encoder = model.transformer.encoder
        self.img_metas = [img_metas]
        # with weights
        self.bev_h = model.bev_h
        self.bev_w = model.bev_w
        self.feat_down_sample_indice = model.transformer.feat_down_sample_indice
        
        self.img_shape = [
            (480, 800, 3), 
            (480, 800, 3),
            (480, 800, 3), 
            (480, 800, 3), 
            (480, 800, 3), 
            (480, 800, 3)]
        pass
        

    def forward(
            self,
            img_feats
        ):
        
        mlvl_feats = [img_feats]
        mlvl_feats[0] = mlvl_feats[0].float()
        
        img_metas = self.img_metas
        img_metas[0]['img_shape'] = self.img_shape #useful
        
        # 1.1 get bev feats --> lss_bev_encode (pv2bev)
        images = mlvl_feats[self.feat_down_sample_indice] # -1, get the last element with largest resolution
        
        B, N, C, fH, fW = images.shape
        
        encoder_outputdict = self.encoder(images, img_metas)
        bev_embed = encoder_outputdict["bev"]
        depth = encoder_outputdict["depth"]
        bs, c, _, _ = bev_embed.shape # need,comment temporary
        bev_embed = bev_embed.view(bs, c, -1).permute(0, 2, 1).contiguous()
        bev_embed = bev_embed.view(bs, self.bev_h, self.bev_w, -1)
        bev_embed = bev_embed.permute(0, 3, 1, 2).contiguous() # B C H W
            
        return bev_embed, depth


class BEVPooling(nn.Module):
    def __init__(self, model):
        super(BEVPooling, self).__init__()
        self.transformer = model.transformer
        # with weights
        self.bev_h = model.bev_h
        self.bev_w = model.bev_w
        self.feat_down_sample_indice = self.transformer.feat_down_sample_indice
        
        self.img_shape = [
            (480, 800, 3), 
            (480, 800, 3),
            (480, 800, 3), 
            (480, 800, 3), 
            (480, 800, 3), 
            (480, 800, 3)]
        pass
        
    def forward(
            self,
            img_feats,
            lidar_feat,
            img_meta
        ):
        
        mlvl_feats = [img_feats]
        mlvl_feats[0] = mlvl_feats[0].float()
        
        img_metas=[img_meta]
        img_metas[0]['img_shape'] = self.img_shape #useful
        
        # 1.1 get bev feats --> lss_bev_encode (pv2bev) + fuser(with lidar)
        images = mlvl_feats[self.feat_down_sample_indice] # -1, get the last element with largest resolution
        encoder_outputdict = self.transformer.encoder(images, img_metas)
        
        bev_embed = encoder_outputdict["bev"]
        depth = encoder_outputdict["depth"]
        bs, c, _, _ = bev_embed.shape
        bev_embed = bev_embed.view(bs, c, -1).permute(0, 2, 1).contiguous()
        ret_dict = dict(bev=bev_embed, depth=depth)
        
        # (2)fusion
        bev_embed = ret_dict["bev"]
        depth = ret_dict["depth"]
        
        if lidar_feat is not None and self.transformer.fuser is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, self.bev_h, self.bev_w, -1)
            bev_embed = bev_embed.permute(0, 3, 1, 2).contiguous()
        
            lidar_feat = lidar_feat.permute(0, 1, 3, 2).contiguous()  # B C H W
            lidar_feat = nn.functional.interpolate(lidar_feat, size=(self.bev_h, self.bev_w), mode="bicubic", align_corners=False)
            fused_bev = self.transformer.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0, 2, 1).contiguous()
            bev_embed = fused_bev
            
        return bev_embed, depth


class BEVDecoder(nn.Module):
    def __init__(self, model, img_metas):
        super(BEVDecoder, self).__init__()
        self.model = model
        self.img_metas = [img_metas]
        # with weights
        self.reg_branches=self.model.reg_branches if self.model.with_box_refine else None
        self.cls_branches=self.model.cls_branches if self.model.as_two_stage else None
        self.pts_query = self.model.pts_embedding.weight.unsqueeze(0).detach()
        
        self.z_cfg = self.model.z_cfg
        self.aux_seg = self.model.aux_seg
        
        # hyper-parameters
        self.num_vec_one2one = self.model.num_vec_one2one
        self.num_pts_per_vec = self.model.num_pts_per_vec
        
        self.bev_h = self.model.bev_h
        self.bev_w = self.model.bev_w
        self.grid_length=(self.model.real_h / self.model.bev_h, self.model.real_w / self.model.bev_w)
        
        self.feat_down_sample_indice = self.model.transformer.feat_down_sample_indice
        
        self.debug = {}
        
        pass  
    
        
    def forward(
            self,
            img_feats,
            bev_embed,
            depth,
            **kwargs,
        ):
        
        mlvl_feats = [img_feats]
        mlvl_feats[0] = mlvl_feats[0].float()
        
        img_metas = self.img_metas
        
        ret_dict = dict(bev=bev_embed, depth=depth)
        
        bev_embed = ret_dict["bev"]
        depth = ret_dict["depth"]
        
        bs = mlvl_feats[0].size(0)
        
        device = mlvl_feats[0].device
        num_vec = self.num_vec_one2one
        
        self_attn_mask = torch.zeros((num_vec, num_vec), dtype=torch.float32, device=device)
        self_attn_mask[self.num_vec_one2one:, :self.num_vec_one2one] = 1.0
        self_attn_mask[:self.num_vec_one2one, self.num_vec_one2one:] = 1.0
        self_attn_mask = self_attn_mask > 0.5    
        kwargs["self_attn_mask"] = self_attn_mask #good


        # return img_feats
        (mask_pred_list, cls_pred_list, mask_aware_query_feat, bev_embed, bev_embed_ms, depth, dn_information) = \
        self.model.transformer.IMPNet_part(
            ouput_dic=ret_dict,
        )

        # self.debug["mask_pred_list"] = mask_pred_list[0]
        # self.debug["cls_pred_list"] = cls_pred_list[0]
        # self.debug["mask_aware_query_feat"] = mask_aware_query_feat
        # self.debug["bev_embed_ms"] = bev_embed_ms[0]
        # self.debug["dn_information"] = dn_information[0]
        
        # MMPnet
        # 2 call self.PositionalQueryGenerator
        pts_query = self.pts_query
        mask_dict, known_masks, known_bid, map_known_indice = dn_information
        mask_aware_query_feat, mask_aware_query_pos = self.model.transformer.PositionalQueryGenerator(
            mask_aware_query_feat,
            pts_query,
            mask_pred_list[-1],
            known_masks=known_masks,
            known_bid=known_bid,
            map_known_indice=map_known_indice,
        )
        
        # self.debug["mask_aware_query_feat1"] = mask_aware_query_feat
        # self.debug["mask_aware_query_pos"] = mask_aware_query_pos
        
        # 3 call self.GeometricFeatureExtractor
        
        # self.debug["mask_pred_list_1"] = mask_pred_list[0]
        # self.debug["mask_aware_query_feat_1"] = mask_aware_query_feat
        # self.debug["mask_aware_query_pos_1"] = mask_aware_query_pos
        # self.debug["bev_embed_1"] = bev_embed
        
        query, key, value, mask_aware_query, mask_aware_query_norm = self.model.transformer.GeometricFeatureExtractor(
            mask_pred_list,
            mask_aware_query_feat,
            mask_aware_query_pos,
            bev_embed,
        )
        
        # self.debug.update(self.model.transformer.debug)
        # self.debug["query"] = query
        # self.debug["key"] = key
        # self.debug["value"] = value
        # self.debug["mask_aware_query"] = mask_aware_query
        # self.debug["mask_aware_query_norm"] = mask_aware_query_norm
        
        # 4 call self.MaskGuidedMapDecoder
        kwargs["num_vec"] = num_vec
        kwargs["num_pts_per_vec"] = self.num_pts_per_vec
        
        inter_states, inter_references_out, init_reference_out = self.model.transformer.MaskGuidedMapDecoder(
            query,
            key,
            value,
            bev_embed_ms,
            mask_aware_query,
            mask_aware_query_norm,
            self.reg_branches,
            self.cls_branches,
            img_metas=img_metas,
            **kwargs,
        )
        
        # self.debug["mask_aware_query1"] = mask_aware_query
        # self.debug["mask_aware_query_norm1"] = mask_aware_query_norm
        # self.debug["inter_states"] = inter_states
        # self.debug["inter_references_out"] = inter_references_out
        # self.debug["init_reference_out"] = init_reference_out

        outputs = bev_embed, mask_pred_list, cls_pred_list, depth, inter_states, init_reference_out, inter_references_out, mask_dict

        bev_embed, segm_mask_pred_list, segm_cls_scores_list, depth, hs, init_reference, inter_references, mask_dict = outputs
        hs = hs.permute(0, 2, 1, 3)
        dn_pad_size = mask_dict["pad_size"] if mask_dict is not None else 0

        outputs_segm_mask_pred_dn = []
        outputs_segm_cls_scores_dn = []

        outputs_segm_mask_pred_one2one = []
        outputs_segm_cls_scores_one2one = []

        outputs_segm_mask_pred_one2many = []
        outputs_segm_cls_scores_one2many = []

        for lvl in range(len(segm_mask_pred_list)):
            outputs_segm_mask_pred_dn.append(segm_mask_pred_list[lvl][:, :dn_pad_size])
            outputs_segm_cls_scores_dn.append(segm_cls_scores_list[lvl][:, :dn_pad_size])

            outputs_segm_mask_pred_one2one.append(segm_mask_pred_list[lvl][:, dn_pad_size : dn_pad_size + self.num_vec_one2one])
            outputs_segm_cls_scores_one2one.append(segm_cls_scores_list[lvl][:, dn_pad_size : dn_pad_size + self.num_vec_one2one])

            outputs_segm_mask_pred_one2many.append(segm_mask_pred_list[lvl][:, dn_pad_size + self.num_vec_one2one :])
            outputs_segm_cls_scores_one2many.append(segm_cls_scores_list[lvl][:, dn_pad_size + self.num_vec_one2one :])

        outputs_segm_mask_pred_dn = torch.stack(outputs_segm_mask_pred_dn)
        outputs_segm_cls_scores_dn = torch.stack(outputs_segm_cls_scores_dn)
        outputs_segm_mask_pred_one2one = torch.stack(outputs_segm_mask_pred_one2one)
        outputs_segm_cls_scores_one2one = torch.stack(outputs_segm_cls_scores_one2one)
        outputs_segm_mask_pred_one2many = torch.stack(outputs_segm_mask_pred_one2many)
        outputs_segm_cls_scores_one2many = torch.stack(outputs_segm_cls_scores_one2many)

        outputs_classes_dn = []
        outputs_coords_dn = []
        outputs_pts_coords_dn = []

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_pts_coords_one2one = []

        outputs_classes_one2many = []
        outputs_coords_one2many = []
        outputs_pts_coords_one2many = []

        num_vec = hs.shape[2] // self.num_pts_per_vec
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # import pdb;pdb.set_trace()
                reference = init_reference[..., 0:2] if not self.z_cfg["gt_z_flag"] else init_reference[..., 0:3]
            else:
                reference = inter_references[lvl - 1][..., 0:2] if not self.z_cfg["gt_z_flag"] else inter_references[lvl - 1][..., 0:3]
            reference = inverse_sigmoid(reference)
            outputs_class = self.model.cls_branches[lvl](hs[lvl].view(bs, num_vec, self.num_pts_per_vec, -1).mean(2))
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp = tmp[..., 0:2] if not self.z_cfg["gt_z_flag"] else tmp[..., 0:3]
            tmp += reference

            tmp = tmp.sigmoid()  # cx,cy,w,h

            outputs_coord, outputs_pts_coord = self.model.transform_box(tmp, num_vec=num_vec)

            outputs_classes_one2one.append(outputs_class[:, dn_pad_size : dn_pad_size + self.num_vec_one2one])
            outputs_coords_one2one.append(outputs_coord[:, dn_pad_size : dn_pad_size + self.num_vec_one2one])
            outputs_pts_coords_one2one.append(outputs_pts_coord[:, dn_pad_size : dn_pad_size + self.num_vec_one2one])

            outputs_classes_dn.append(outputs_class[:, :dn_pad_size])
            outputs_coords_dn.append(outputs_coord[:, :dn_pad_size])
            outputs_pts_coords_dn.append(outputs_pts_coord[:, :dn_pad_size])

            outputs_classes_one2many.append(outputs_class[:, dn_pad_size + self.num_vec_one2one :])
            outputs_coords_one2many.append(outputs_coord[:, dn_pad_size + self.num_vec_one2one :])
            outputs_pts_coords_one2many.append(outputs_pts_coord[:, dn_pad_size + self.num_vec_one2one :])

        outputs_classes_dn = torch.stack(outputs_classes_dn)
        outputs_coords_dn = torch.stack(outputs_coords_dn)
        outputs_pts_coords_dn = torch.stack(outputs_pts_coords_dn)

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)

        if self.model.dn_enabled and self.model.training:
            mask_dict["output_known_lbs_bboxes"] = (
                outputs_segm_cls_scores_dn,
                outputs_segm_mask_pred_dn,
                outputs_classes_dn,
                outputs_coords_dn,
                outputs_pts_coords_dn,
            )

        outputs_seg = None
        outputs_pv_seg = None
        if self.aux_seg["use_aux_seg"]:
            seg_bev_embed = \
                bev_embed.reshape(bs, self.model.bev_h, self.model.bev_w, -1).permute(0, 3, 1, 2).contiguous()
            if self.aux_seg["bev_seg"]:
                outputs_seg = self.model.seg_head(seg_bev_embed)
            bs, num_cam, embed_dims, feat_h, feat_w = mlvl_feats[-1].shape
            if self.aux_seg["pv_seg"]:
                outputs_pv_seg = self.model.pv_seg_head(mlvl_feats[-1].flatten(0, 1))
                outputs_pv_seg = outputs_pv_seg.view(bs, num_cam, -1, feat_h, feat_w)

        if self.model.tr_cls_join and self.model.training:
            outputs_classes_one2one = outputs_classes_one2one + outputs_segm_cls_scores_one2one[-1:][..., :3].contiguous()

        if self.model.cls_join and not self.model.training:
            outputs_classes_one2one = outputs_classes_one2one + outputs_segm_cls_scores_one2one[-1:][..., :3].contiguous()
        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes_one2one, # used to bbox decode
            "all_bbox_preds": outputs_coords_one2one, # used to bbox decode
            "all_pts_preds": outputs_pts_coords_one2one, # used to bbox decode
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
            "depth": depth,
            "seg": outputs_seg,
            "segm_mask_preds": outputs_segm_mask_pred_one2one,
            "segm_cls_scores": outputs_segm_cls_scores_one2one,
            "pv_seg": outputs_pv_seg,
            "one2many_outs": dict(
                all_cls_scores=outputs_classes_one2many,
                all_bbox_preds=outputs_coords_one2many,
                all_pts_preds=outputs_pts_coords_one2many,
                enc_bbox_preds=None,
                enc_pts_preds=None,
                segm_mask_preds=outputs_segm_mask_pred_one2many,
                segm_cls_scores=outputs_segm_cls_scores_one2many,
                seg=None,
                pv_seg=None,
            ),
            "dn_mask_dict": mask_dict,
            "debug": self.debug,
        }

        return self.get_bboxes_whole(outs, img_metas)
    
    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes_whole(self, preds_dicts, img_metas):
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.model.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]
            pts = preds["pts"]

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list
    
    def forward_trt(
            self,
            # img_feats,
            bev_embed,
            # depth,
            **kwargs,
        ):
        
        depth = None
        img_metas = self.img_metas
        
        ret_dict = dict(bev=bev_embed, depth=depth)
        
        bs = bev_embed.size(0)
        device = bev_embed.device
        num_vec = self.num_vec_one2one
        
        self_attn_mask = torch.zeros((num_vec, num_vec), dtype=torch.float32, device=device)
        self_attn_mask[self.num_vec_one2one:, :self.num_vec_one2one] = 1.0
        self_attn_mask[:self.num_vec_one2one, self.num_vec_one2one:] = 1.0
        self_attn_mask = self_attn_mask > 0.5
        kwargs["self_attn_mask"] = self_attn_mask #good
        
        (mask_pred_list, cls_pred_list, mask_aware_query_feat, bev_embed, bev_embed_ms, depth, dn_information) = \
        self.model.transformer.IMPNet_part(
            ouput_dic=ret_dict,
        ) #if close attention, onnx good
        
        # if we can get mask_pred_list, then I can simply use post-process to vectorize the mask_pred_list
        
        # MMPnet
        # 2 call self.PositionalQueryGenerator
        pts_query = self.pts_query
        mask_dict, known_masks, known_bid, map_known_indice = dn_information
        mask_aware_query_feat, mask_aware_query_pos = self.model.transformer.PositionalQueryGenerator(
            mask_aware_query_feat,
            pts_query,
            mask_pred_list[-1],
            known_masks=known_masks,
            known_bid=known_bid,
            map_known_indice=map_known_indice,
        ) # almost good
        
        
        # 3 call self.GeometricFeatureExtractor
        query, key, value, mask_aware_query, mask_aware_query_norm = self.model.transformer.GeometricFeatureExtractor(
            mask_pred_list,
            mask_aware_query_feat,
            mask_aware_query_pos,
            bev_embed,
        ) # almost good
    
        # 4 call self.MaskGuidedMapDecoder
        kwargs["num_vec"] = num_vec
        kwargs["num_pts_per_vec"] = self.num_pts_per_vec
        
        inter_states, inter_references_out, init_reference_out = self.model.transformer.MaskGuidedMapDecoder(
            query,
            key,
            value,
            bev_embed_ms,
            mask_aware_query,
            mask_aware_query_norm,
            self.reg_branches,
            self.cls_branches,
            img_metas=img_metas,
            **kwargs,
        ) # almost good, closs self-attn & cross-attn

        outputs = bev_embed, mask_pred_list, cls_pred_list, depth, inter_states, init_reference_out, inter_references_out, mask_dict

        bev_embed, segm_mask_pred_list, segm_cls_scores_list, depth, hs, init_reference, inter_references, mask_dict = outputs
        hs = hs.permute(0, 2, 1, 3)
        dn_pad_size = mask_dict["pad_size"] if mask_dict is not None else 0

        outputs_segm_mask_pred_dn = []
        outputs_segm_cls_scores_dn = []

        outputs_segm_mask_pred_one2one = []
        outputs_segm_cls_scores_one2one = []

        outputs_segm_mask_pred_one2many = []
        outputs_segm_cls_scores_one2many = []

        for lvl in range(len(segm_mask_pred_list)):
            outputs_segm_mask_pred_dn.append(segm_mask_pred_list[lvl][:, :dn_pad_size])
            outputs_segm_cls_scores_dn.append(segm_cls_scores_list[lvl][:, :dn_pad_size])

            outputs_segm_mask_pred_one2one.append(segm_mask_pred_list[lvl][:, dn_pad_size : dn_pad_size + self.num_vec_one2one])
            outputs_segm_cls_scores_one2one.append(segm_cls_scores_list[lvl][:, dn_pad_size : dn_pad_size + self.num_vec_one2one])

            outputs_segm_mask_pred_one2many.append(segm_mask_pred_list[lvl][:, dn_pad_size + self.num_vec_one2one :])
            outputs_segm_cls_scores_one2many.append(segm_cls_scores_list[lvl][:, dn_pad_size + self.num_vec_one2one :])

        outputs_segm_mask_pred_dn = torch.stack(outputs_segm_mask_pred_dn)
        outputs_segm_cls_scores_dn = torch.stack(outputs_segm_cls_scores_dn)
        outputs_segm_mask_pred_one2one = torch.stack(outputs_segm_mask_pred_one2one)
        outputs_segm_cls_scores_one2one = torch.stack(outputs_segm_cls_scores_one2one)
        outputs_segm_mask_pred_one2many = torch.stack(outputs_segm_mask_pred_one2many)
        outputs_segm_cls_scores_one2many = torch.stack(outputs_segm_cls_scores_one2many)

        outputs_classes_dn = []
        outputs_coords_dn = []
        outputs_pts_coords_dn = []

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_pts_coords_one2one = []

        outputs_classes_one2many = []
        outputs_coords_one2many = []
        outputs_pts_coords_one2many = []
        
        num_vec = hs.shape[2] // self.num_pts_per_vec
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # import pdb;pdb.set_trace()
                reference = init_reference[..., 0:2] if not self.z_cfg["gt_z_flag"] else init_reference[..., 0:3]
            else:
                reference = inter_references[lvl - 1][..., 0:2] if not self.z_cfg["gt_z_flag"] else inter_references[lvl - 1][..., 0:3]
            reference = inverse_sigmoid(reference)
            outputs_class = self.model.cls_branches[lvl](hs[lvl].view(bs, num_vec, self.num_pts_per_vec, -1).mean(2))
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp = tmp[..., 0:2] if not self.z_cfg["gt_z_flag"] else tmp[..., 0:3]
            tmp += reference

            tmp = tmp.sigmoid()  # cx,cy,w,h

            outputs_coord, outputs_pts_coord = self.model.transform_box(tmp, num_vec=num_vec)

            outputs_classes_one2one.append(outputs_class[:, dn_pad_size : dn_pad_size + self.num_vec_one2one])
            outputs_coords_one2one.append(outputs_coord[:, dn_pad_size : dn_pad_size + self.num_vec_one2one])
            outputs_pts_coords_one2one.append(outputs_pts_coord[:, dn_pad_size : dn_pad_size + self.num_vec_one2one])

            outputs_classes_dn.append(outputs_class[:, :dn_pad_size])
            outputs_coords_dn.append(outputs_coord[:, :dn_pad_size])
            outputs_pts_coords_dn.append(outputs_pts_coord[:, :dn_pad_size])

            outputs_classes_one2many.append(outputs_class[:, dn_pad_size + self.num_vec_one2one :])
            outputs_coords_one2many.append(outputs_coord[:, dn_pad_size + self.num_vec_one2one :])
            outputs_pts_coords_one2many.append(outputs_pts_coord[:, dn_pad_size + self.num_vec_one2one :])

        outputs_classes_dn = torch.stack(outputs_classes_dn)
        outputs_coords_dn = torch.stack(outputs_coords_dn)
        outputs_pts_coords_dn = torch.stack(outputs_pts_coords_dn)

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)

        if self.model.dn_enabled and self.model.training:
            mask_dict["output_known_lbs_bboxes"] = (
                outputs_segm_cls_scores_dn,
                outputs_segm_mask_pred_dn,
                outputs_classes_dn,
                outputs_coords_dn,
                outputs_pts_coords_dn,
            )

        outputs_seg = None
        if self.aux_seg["use_aux_seg"] and self.aux_seg["bev_seg"]:
            seg_bev_embed = \
                bev_embed.reshape(bs, self.model.bev_h, self.model.bev_w, -1).permute(0, 3, 1, 2).contiguous()
            outputs_seg = self.model.seg_head(seg_bev_embed)
        
        outputs_pv_seg = None
        # mlvl_feats = [img_feats]
        # mlvl_feats[0] = mlvl_feats[0].float()
        # if self.aux_seg["use_aux_seg"]:
        #     bs, num_cam, embed_dims, feat_h, feat_w = mlvl_feats[-1].shape
        #     if self.aux_seg["pv_seg"]:
        #         outputs_pv_seg = self.model.pv_seg_head(mlvl_feats[-1].flatten(0, 1))
        #         outputs_pv_seg = outputs_pv_seg.view(bs, num_cam, -1, feat_h, feat_w)

        if self.model.tr_cls_join and self.model.training:
            outputs_classes_one2one = outputs_classes_one2one + outputs_segm_cls_scores_one2one[-1:][..., :3].contiguous()

        if self.model.cls_join and not self.model.training:
            outputs_classes_one2one = outputs_classes_one2one + outputs_segm_cls_scores_one2one[-1:][..., :3].contiguous()
        
        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes_one2one, # used to bbox decode
            "all_bbox_preds": outputs_coords_one2one, # used to bbox decode
            "all_pts_preds": outputs_pts_coords_one2one, # used to bbox decode
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
            # "depth": depth,
            "depth": None,
            "seg": outputs_seg,
            "segm_mask_preds": outputs_segm_mask_pred_one2one,
            "segm_cls_scores": outputs_segm_cls_scores_one2one,
            "pv_seg": outputs_pv_seg,
            "one2many_outs": dict(
                all_cls_scores=outputs_classes_one2many,
                all_bbox_preds=outputs_coords_one2many,
                all_pts_preds=outputs_pts_coords_one2many,
                enc_bbox_preds=None,
                enc_pts_preds=None,
                segm_mask_preds=outputs_segm_mask_pred_one2many,
                segm_cls_scores=outputs_segm_cls_scores_one2many,
                seg=None,
                pv_seg=None,
            ),
            "dn_mask_dict": mask_dict,
            "debug": self.debug,
        }
    
        return outputs_classes_one2one, outputs_coords_one2one, outputs_pts_coords_one2one, outputs_seg

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, classes, coords, pts_coords):
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = {}
        preds_dicts["all_cls_scores"] = classes
        preds_dicts["all_bbox_preds"] = coords
        preds_dicts["all_pts_preds"] = pts_coords

        preds_dicts = self.model.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            scores = preds["scores"]
            labels = preds["labels"]
            pts = preds["pts"]

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list