from typing import List
import os
import argparse
import json
import random
import time
from pathlib import Path
import sys
import numpy as np
import torch
import util.misc as utils
import io
import scipy.ndimage as ndimage
import datasets_inference.transforms as T

from tqdm import tqdm

from PIL import Image
from util.slconfig import DictAction, SLConfig
from util.misc import nested_tensor_from_tensor_list
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from groundingdino.util.box_ops import box_cxcywh_to_xyxy

# from vision_agent.tools import overlay_bounding_boxes


CHECKPOINT = "../../CountGD/checkpoints/countgd.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESH = 0.23
CFG_FILE = "config/cfg_fsc147_test.py"
_SEED = 49


def get_args_parser():

    parser = argparse.ArgumentParser("CountGD Argument Parser", add_help=False)
    # parser.add_argument("--config_file", "-c", type=str, required=True)
    # parser.add_argument(
    #     "--options",
    #     nargs="+",
    #     action=DictAction,
    #     help="override some settings in the used config, the key-value pair "
    #     "in xxx=yyy format will be merged into config file.",
    # )

    # # dataset parameters
    # parser.add_argument(
    #     "--datasets", type=str, required=False, help="path to datasets json"
    # )
    # parser.add_argument("--no_text", action="store_true")
    # parser.add_argument(
    #     "--num_exemplars", default=3, type=int, help="number of visual exemplars to use"
    # )
    # parser.add_argument("--remove_difficult", action="store_true")
    # parser.add_argument("--fix_size", action="store_true")

    # training parameters
    # parser.add_argument(
    #     "--train_with_exemplar_only",
    #     action="store_true",
    #     help="train with only exemplars",
    # )
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    # parser.add_argument("--note", default="", help="add some notes to the experiment")
    # parser.add_argument(
    #     "--device", default="cuda", help="device to use for training / testing"
    # )
    # parser.add_argument("--seed", default=42, type=int)
    # parser.add_argument("--resume", default="", help="resume from checkpoint")
    # parser.add_argument("--pretrain_model_path", help="load from other checkpoint")
    # parser.add_argument("--finetune_ignore", type=str, nargs="+")
    # parser.add_argument(
    #     "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    # )
    # parser.add_argument("--eval", action="store_true")
    # parser.add_argument("--num_workers", default=8, type=int)
    # parser.add_argument("--test", action="store_true")
    # parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--find_unused_params", action="store_true")
    # parser.add_argument("--save_results", action="store_true")
    # parser.add_argument("--save_log", action="store_true")
    # parser.add_argument(
    #     "--modality_dropout",
    #     help="randomly drop all text or all exemplars during training",
    #     action="store_true",
    # )

    # evaluation parameters
    # parser.add_argument(
    #     "--sam_tt_norm",
    #     action="store_true",
    #     help="apply test-time normalization using Segment Anything Model (SAM) for refinement and visual exemplars as box prompts",
    # )
    # parser.add_argument(
    #     "--sam_model_path",
    #     default="./checkpoints/sam_vit_h_4b8939.pth",
    #     help="path to SAM model checkpoint",
    # )
    # parser.add_argument(
    #     "--exemp_tt_norm",
    #     action="store_true",
    #     help="apply test-time normalization using the visual exemplars (no SAM) only",
    # )
    # parser.add_argument(
    #     "--crop",
    #     action="store_true",
    #     help="use visual exemplars to adaptively crop images with max number of objects (i.e., [num_select], e.g., 900) detected into smaller pieces for final count prediction",
    # )
    # parser.add_argument(
    #     "--simple_crop",
    #     action="store_true",
    #     help="No exemplar version of --crop. Crop images with max number of objects (i.e., [num_select], e.g., 900) detected into smaller pieces for final count prediction",
    # )
    # parser.add_argument(
    #     "--remove_bad_exemplar",
    #     action="store_true",
    #     help="remove inaccurate exemplar annotations and use text instead",
    # )
    parser.add_argument(
        "--prompts",
        type=json.loads,
        help="A list of lists in JSON format",
        default=None,
    )
    parser.add_argument(
        "--text",
        type=str,
        help="A text indicating the object",
        default="green lid",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="the image to process",
        default="/home/shankar/repos/vision-agent-benchmark/data/vbox.jpg",
    )

    # distributed training parameters
    # parser.add_argument(
    #     "--world_size", default=1, type=int, help="number of distributed processes"
    # )
    # parser.add_argument(
    #     "--dist_url", default="env://", help="url used to set up distributed training"
    # )
    # parser.add_argument(
    #     "--rank", default=0, type=int, help="number of distributed processes"
    # )
    # parser.add_argument(
    #     "--local_rank", type=int, help="local rank for DistributedDataParallel"
    # )
    # parser.add_argument(
    #     "--local-rank", type=int, help="local rank for DistributedDataParallel"
    # )
    # parser.add_argument("--amp", action="store_true", help="Train with mixed precision")
    return parser


def build_model_main(cfg):
    # we use register to maintain models from catdet6 on.
    from models_inference.registry import MODULE_BUILD_FUNCS

    assert cfg.modelname in MODULE_BUILD_FUNCS._module_dict
    # Add required args to cfg
    cfg.device = DEVICE
    cfg.text_encoder_type = "../../CountGD/checkpoints/bert-base-uncased"
    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    model, criterion, postprocessors = build_func(cfg)
    return model, criterion, postprocessors


def get_ind_to_filter(text, word_ids, keywords):
    if len(keywords) <= 0:
        return list(range(len(word_ids)))
    input_words = text.split()
    keywords = keywords.split(",")
    keywords = [keyword.strip() for keyword in keywords]

    word_inds = []
    for keyword in keywords:
        if keyword in input_words:
            if len(word_inds) <= 0:
                ind = input_words.index(keyword)
                word_inds.append(ind)
            else:
                ind = input_words.index(keyword, word_inds[-1])
                word_inds.append(ind)
        else:
            raise Exception("Only specify keywords in the input text!")

    inds_to_filter = []
    for ind in range(len(word_ids)):
        word_id = word_ids[ind]
        if word_id in word_inds:
            inds_to_filter.append(ind)

    return inds_to_filter


def infer(args):
    # load cfg file and update the args
    print("Loading config file from {}".format(CFG_FILE))
    cfg = SLConfig.fromfile(CFG_FILE)
    # if args.options is not None:
    #     cfg.merge_from_dict(args.options)
    # if args.rank == 0:
    #     save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
    #     cfg.dump(save_cfg_path)
    #     save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
    #     with open(save_json_path, "w") as f:
    #         json.dump(vars(args), f, indent=2)
    # cfg_dict = cfg._cfg_dict.to_dict()
    # args_vars = vars(args)
    # for k, v in cfg_dict.items():
    #     if k not in args_vars:
    #         setattr(args, k, v)
    #     else:
    #         raise ValueError("Key {} can used by args only".format(k))

    # setup output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # setup seed
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)
    random.seed(_SEED)

    # build model
    print("building model ... ...")
    model, _, _ = build_model_main(cfg)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval().to(DEVICE)
    print("build model, done.")

    # build pre-processing transforms
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    data_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            normalize,
        ]
    )

    count, dmap_img = count_and_visualize(
        model,
        Image.open(args.image).convert("RGB"),
        args.text,
        {"points": args.prompts} if args.prompts is not None else None,
        data_transform,
    )
    print("Total number of instances: ", count)
    dmap_img.save(os.path.join(args.output_dir, "dmap.png"))


def count_and_visualize(model, image, text, prompts, transform):
    keywords = ""  # do not handle this for now
    # Handle no prompt case.
    if prompts is None:
        prompts = {"image": image, "points": []}
    else:
        prompts["image"] = image
    input_image, _ = transform(image, {"exemplars": torch.tensor([])})
    input_image = input_image.unsqueeze(0).to(DEVICE)
    exemplars = prompts["points"]

    input_image_exemplars, exemplars = transform(
        prompts["image"], {"exemplars": torch.tensor(exemplars)}
    )
    input_image_exemplars = input_image_exemplars.unsqueeze(0).to(DEVICE)
    exemplars = [exemplars["exemplars"].to(DEVICE)]

    with torch.no_grad():
        model_output = model(
            nested_tensor_from_tensor_list(input_image),
            nested_tensor_from_tensor_list(input_image_exemplars),
            exemplars,
            [torch.tensor([0]).to(DEVICE) for _ in range(len(input_image))],
            captions=[text + " ."] * len(input_image),
        )

    ind_to_filter = get_ind_to_filter(text, model_output["token"][0].word_ids, keywords)
    logits = model_output["pred_logits"].sigmoid()[0][:, ind_to_filter]
    boxes = model_output["pred_boxes"][0]

    if len(keywords.strip()) > 0:
        box_mask = (logits > CONF_THRESH).sum(dim=-1) == len(ind_to_filter)
    else:
        box_mask = logits.max(dim=-1).values > CONF_THRESH
    logits = logits[box_mask, :].cpu().numpy()
    boxes_xyxy = box_cxcywh_to_xyxy(boxes[box_mask, :]).cpu().numpy()
    boxes = boxes[box_mask, :].cpu().numpy()

    # create output structure required by va
    assert len(boxes_xyxy) == len(logits)
    result = []
    labels = text.split(" .") if text.strip() != "" else ["object"]

    for i in range(len(boxes_xyxy)):
        if len(labels) == 1:
            lbl = labels[0]
        else:
            lbl_token_ids = [model.tokenizer(x)["input_ids"] for x in labels]
            pred_token_id = model_output["token"][0].ids[int(logits[i].argmax())]
            for i, lbl_token_id in enumerate(lbl_token_ids):
                if pred_token_id in lbl_token_id:
                    lbl = labels[i]
                    break
        result.append(
            {
                "bbox": boxes_xyxy[i],
                "score": float(logits[i].max()),
                "label": lbl,
            }
        )
    # output_img = overlay_bounding_boxes(np.array(image), result)
    # output_img = Image.fromarray(output_img)

    # Plot boxes and dmap.
    plt.imshow(image)
    (w, h) = image.size
    # for box in boxes:
    #     x = int((box[0] - box[2] / 2) * w)
    #     y = int((box[1] - box[3] / 2) * h)
    #     w_s = int(box[2] * w)
    #     h_s = int(box[3] * h)
    #     plt.gca().add_patch(
    #         Rectangle(
    #             (x, y),
    #             w_s,
    #             h_s,
    #             edgecolor="red",
    #             facecolor="none",
    #             lw=1,
    #         )
    #     )
    det_map = np.zeros((h, w))
    det_map[(h * boxes[:, 1]).astype(int), (w * boxes[:, 0]).astype(int)] = 1
    det_map = ndimage.gaussian_filter(det_map, sigma=(w // 200, w // 200), order=0)
    plt.imshow(
        det_map[None, :].transpose(1, 2, 0), "jet", interpolation="none", alpha=0.7
    )
    plt.axis("off")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight")
    plt.close()
    output_img = Image.open(img_buf)

    out_label = "Detected instances predicted with"
    if len(text.strip()) > 0:
        out_label += " text"
        if exemplars[0].size()[0] == 1:
            out_label += " and " + str(exemplars[0].size()[0]) + " visual exemplar."
        elif exemplars[0].size()[0] > 1:
            out_label += " and " + str(exemplars[0].size()[0]) + " visual exemplars."
        else:
            out_label += "."
    elif exemplars[0].size()[0] > 0:
        if exemplars[0].size()[0] == 1:
            out_label += " " + str(exemplars[0].size()[0]) + " visual exemplar."
        else:
            out_label += " " + str(exemplars[0].size()[0]) + " visual exemplars."
    else:
        out_label = "Nothing specified to detect."

    print(out_label)
    return len(result), output_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "CountGD training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    infer(args)
