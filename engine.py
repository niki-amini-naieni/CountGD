# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import numpy as np
import math
import random
import os
import sys
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from util.utils import to_device
from util.visualizer import renorm
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from datasets.panoptic_eval import PanopticEvaluator
#from skimage.filters import threshold_otsu


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        exemplars = [t["exemplars"].to(device) for t in targets]
        labels_uncropped = [t["labels_uncropped"].to(device) for t in targets]
        min_exemplars_in_batch = min([exemp.shape[0] for exemp in exemplars])
        shot_num = min(random.randint(0, 3), min_exemplars_in_batch)
        # REMOVE WHEN TRYING DIFFERENT NUMBERS OF VISUAL EXEMPLARS.
        shot_num = min_exemplars_in_batch
        # Adjust number of exemplars based on [shot_num].
        exemplars = [exemp[:shot_num] for exemp in exemplars]
        for exemp in exemplars:
            if exemp.shape[0] > 3:
                print('WARNING: Exemp shape greater than 3!!! Only 3 exemplars allowed during training')
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, exemplars, labels_uncropped, captions=captions)
            loss_dict = criterion(outputs, targets, cap_list, captions)

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat

def plot_points(image, exemplars, size, points):
    (h, w) = (size[0], size[1])
    for point in points:
        in_exemp = (point[0] * w > exemplars[:, 0]) * (point[0] * w < exemplars[:, 2])
        in_exemp = (in_exemp) * (point[1] * h > exemplars[:, 1]) * (point[1] * h < exemplars[:, 3])
        if in_exemp.sum() > 0:
            plt.plot(point[0] * w, point[1] * h, marker='v', color="red")
        else:
            plt.plot(point[0] * w, point[1] * h, marker='v', color="white")
    for exemp in exemplars:
        plt.gca().add_patch(Rectangle((exemp[0], exemp[1]), exemp[2] - exemp[0], exemp[3] - exemp[1], edgecolor='red', facecolor='none', lw=4))
    plt.imshow(image)
    plt.savefig("sunglasses")

def tt_norm(pred_cnt, exemplars, size, points):
    e_cnt = 0
    (h, w) = (size[0], size[1])
    for point in points:
        in_exemp = (point[0] * w > exemplars[:, 0]) * (point[0] * w < exemplars[:, 2])
        in_exemp = (in_exemp) * (point[1] * h > exemplars[:, 1]) * (point[1] * h < exemplars[:, 3])
        if in_exemp.sum() > 0:
            e_cnt += 1
    e_cnt = e_cnt / exemplars.shape[0]
    if e_cnt >= (5 / 3):
        # At least 2 of the exemplars contain 2 object instances.
        pred_cnt = pred_cnt / e_cnt
    return pred_cnt

def get_count_errs(samples, exemplars, outputs, box_threshold, text_threshold, targets, tokenized_captions, input_captions):
    logits = outputs['pred_logits'].sigmoid()
    boxes = outputs['pred_boxes']
    np.save("logits.npy", logits.cpu().numpy())
    samples = samples.to_img_list()
    sizes = [target["size"] for target in targets]
    
    abs_errs = []
    for sample_ind in range(len(targets)):
        sample_logits = logits[sample_ind]
        sample_boxes = boxes[sample_ind]
        input_caption = input_captions[sample_ind]
        sample = samples[sample_ind]
        size = sizes[sample_ind]
        sample_exemplars = exemplars[sample_ind]

        # Setting adaptive logit threshold based on Otsu's binarization algo.
        #max_logits = sample_logits.max(dim=-1).values.cpu().numpy()
        #box_threshold = threshold_otsu(max_logits)

        for token_ind in range(len(tokenized_captions['input_ids'][sample_ind])):
            idx = tokenized_captions['input_ids'][sample_ind][token_ind]
            print(idx)
            if idx == 1012:
                end_idx = token_ind
                break

        box_mask = sample_logits.max(dim=-1).values > box_threshold
        expected_cnt = sample_logits.max(dim=-1).values.sum().item()
        expected_cnt = sample_logits[:, 1:end_idx].mean(dim=-1).sum().item()
        sample_logits = sample_logits[box_mask, :]
        sample_boxes = sample_boxes[box_mask, :]
        
        text_mask = (sample_logits[:, 1:end_idx] > text_threshold).sum(dim=-1) == (end_idx - 1)
        sample_logits = sample_logits[text_mask, :]
        sample_boxes = sample_boxes[text_mask, :]
        #if 'sunglass' in input_caption:
            #plot_points(renorm(sample.cpu()).permute(1, 2, 0).numpy(), sample_exemplars.cpu().numpy(), size.cpu().numpy(), sample_boxes[:,:2].cpu().numpy())
        
        gt_count = targets[sample_ind]['labels'].shape[0]
        pred_cnt = sample_logits.shape[0]
        #pred_cnt = tt_norm(pred_cnt, sample_exemplars.cpu().numpy(), size.cpu().numpy(), sample_boxes[:, :2].cpu().numpy())
        #pred_cnt = expected_cnt
        #pred_cnt = expected_cnt
        if pred_cnt == 0:
            print("All query logits: " + str(logits[sample_ind]))
            print("First query logit: " + str(logits[sample_ind][0]))
            print("tokenized caption: " + str(tokenized_captions['input_ids']))
        #pred_cnt = expected_cnt
        print("Pred Count: " + str(pred_cnt) + ", GT Count: " + str(gt_count))
        abs_errs.append(np.abs(gt_count - pred_cnt)) 
    return abs_errs

@torch.no_grad()
def evaluate(model, model_without_ddp, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    
    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)


    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    if args.use_coco_eval:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)

        # 获取所有类别
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item['name'] for item in category_dict]
    else:
        cat_list=args.val_label_list
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    abs_errs = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        exemplars = [t["exemplars"].to(device) for t in targets]
        labels = [t["labels"].to(device) for t in targets]
        #exemplars = [torch.tensor([]).to(device) for t in targets]
        
        bs = samples.tensors.shape[0]
        input_captions = [cat_list[target['labels'][0]] + " ." for target in targets]
        print("input_captions: " + str(input_captions))
        with torch.cuda.amp.autocast(enabled=args.amp):

            outputs = model(samples, exemplars, [torch.tensor([0]).to(device) for t in targets], captions=input_captions)

        
        tokenized_captions = outputs["token"]
        abs_errs += get_count_errs(samples, exemplars, outputs, args.box_threshold, args.text_threshold, targets, tokenized_captions, input_captions)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:



            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res['boxes']
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
       

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break
    count_mae = sum(abs_errs) / len(abs_errs)
    count_rmse = (np.array(abs_errs) ** 2).mean() ** (1/2)
    print("# of Images Tested: " + str(len(abs_errs)))
    print("MAE: " + str(count_mae) + ", RMSE: " + str(count_rmse))
    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return count_mae, stats, coco_evaluator


