import torch

from util.clip_utils import build_text_embedding_coco, build_text_embedding_lvis

from .backbone import build_backbone
from .deformable_transformer import build_conditional_transformer
from .matcher import build_matcher, build_joint_matcher
from .model import CLIPDeformableDETR
from .post_process import PostProcessSegm, new_PostProcess, PostProcess_deformable
from .segmentation_bak import DETRsegm
from .set_criterion import SetCriterion, SetCriterion_bak


def build_model(args):
    if args.dataset_file == "coco":
        num_classes = 65
    elif args.dataset_file == "lvis":
        num_classes = 1203
    else:
        raise NotImplementedError

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_conditional_transformer(args)

    if args.dataset_file == "coco":
        zeroshot_w = build_text_embedding_coco()
    elif args.dataset_file == "lvis":
        zeroshot_w = build_text_embedding_lvis()
    else:
        raise NotImplementedError

    print("text_embedding_weight_shape:", zeroshot_w.size())

    model = CLIPDeformableDETR(
        backbone, transformer, num_classes=num_classes, num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels, aux_loss=args.aux_loss, with_box_refine=args.with_box_refine,
        two_stage=args.two_stage, embedding_classifier=zeroshot_w, max_len=args.max_len,
        clip_feat_path=args.clip_feat_path, clip_feat_prob=args.prob,
    )

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    weight_dict["loss_embed"] = args.feature_loss_coef
    weight_dict["loss_contrastive"] = args.contrastive_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "contrastive"] # embed
    #losses = ["labels", "boxes", "contrastive"]
    if args.masks:
        losses = ["labels", "boxes", "contrastive", "masks"]

    criterion = SetCriterion_bak(
        num_classes, matcher, weight_dict, losses, focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
        eos_coef=args.eos_coef, temperature=args.temperature,
        scale_by_temperature=args.scale_by_temperature,
    )

#    postprocessors = {"bbox": OVPostProcess(num_queries=args.num_queries)}
    postprocessors = {"bbox": new_PostProcess()}
    criterion.to(device)

    if args.masks:
        postprocessors["segm"] = PostProcessSegm()

    return model, criterion, postprocessors
