import glob
import random
import torch
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import argparse
import json
from util.slconfig import SLConfig, DictAction
from util.misc import nested_tensor_from_tensor_list
import datasets_inference.transforms as T
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

'''
NEED TO HAVE A SUBJECT SPECIFICATION SO ONLY USES TOKENS FROM SUBJECT WORDS
'''

# MODEL:
def get_args_parser():
    """
    Example eval command:

    >> python main.py --output_dir ./gdino_test -c config/cfg_fsc147_vit_b_test.py --eval --datasets config/datasets_fsc147.json --pretrain_model_path ../checkpoints_and_logs/gdino_train/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased --sam_tt_norm --crop
    """
    parser = argparse.ArgumentParser("Testing on CountBench", add_help=False)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # training parameters
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--pretrain_model_path",
        help="load from other checkpoint",
        default="../checkpoints_and_logs/gdino_train/checkpoint_best_regular.pth",
    )
    parser.add_argument(
        "--config",
        help="config file",
        default="./config/cfg_fsc147_vit_b.py",
    )
    parser.add_argument(
        "--images_path",
        help="folder path with all CountBench images",
        default="../CountBench",
    )
    parser.add_argument(
        "--text_descriptions_path",
        help="path of file with all text descriptions",
        default="./data/CountBench.json",
    )
    parser.add_argument(
        "--confidence_thresh", help="confidence threshold for model", default=0.23, type=float
    )
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_false")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument(
        "--local-rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")
    return parser


def build_model_and_transforms(args):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    data_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            normalize,
        ]
    )
    cfg = SLConfig.fromfile(args.config)
    cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)

    model.to(device)

    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    return model, data_transform


def get_input_images(args):
    images = []
    for f in glob.iglob(args.images_path + "/*"):
        images.append({f.split("/")[-1]: Image.open(f)})
    return images


def get_captions(args):
    with open(args.text_descriptions_path) as f:
        captions = json.load(f)
    return captions


def get_inds_from_tokens_and_keyphrases(tokenizer, tokens, keyphrases):
    inds = []
    for keyphrase in keyphrases:
        tokenized_phrase = tokenizer([keyphrase], padding="longest", return_tensors="pt")[0].tokens[1:-1] # remove CLS and SEP tokens
        print(tokenizer([keyphrase], padding="longest", return_tensors="pt")[0])
        print(tokenized_phrase)
        for ind in range(len(tokens)):
            if tokens[ind: (ind + len(tokenized_phrase))] == tokenized_phrase:
                for sub_ind in range(len(tokenized_phrase)):
                    inds.append(ind + sub_ind)
                break
            
    return inds

parser = argparse.ArgumentParser("Testing Counting Model", parents=[get_args_parser()])
args = parser.parse_args()
model, transform = build_model_and_transforms(args)

images = get_input_images(args)
descriptions = get_captions(args)

abs_errs = []
for image in images:
    f_name = list(image.keys())[0]
    input_image, target = transform(image[f_name], {"exemplars": torch.tensor([])})
    input_image = input_image.cuda()
    input_exemplar = target["exemplars"].cuda()
    input_text = descriptions[f_name]["text"]
    gt_count = descriptions[f_name]["count"]
    with torch.no_grad():
        model_output = model(
            input_image.unsqueeze(0),
            [input_exemplar],
            [torch.tensor([0]).cuda()],
            captions=[input_text + " ."],
        )
    logits = model_output["pred_logits"][0].sigmoid()
    boxes = model_output["pred_boxes"][0]
    # Only keep boxes that meet confidence threshold when compared to specified text tokens.
    tokenized_text = model_output["token"][0]

    tokens = tokenized_text.tokens
    ind_to_use = get_inds_from_tokens_and_keyphrases(model.tokenizer, tokens, descriptions[f_name]["keyphrases"])
    print(tokens)
    print(ind_to_use)
    # Only apply confidence threshold to selected text tokens.
    logits = logits[:, ind_to_use]
    box_mask = (logits > args.confidence_thresh).sum(dim=-1) == len(ind_to_use)
    logits = logits[box_mask, :]
    boxes = boxes[box_mask, :]

    pred_count = boxes.shape[0]

    print("Pred. Count: " + str(pred_count) + ", GT Count: " + str(gt_count))
    err = abs(pred_count - gt_count)

    if err <= 0:
        (w, h) = image[f_name].size
        det_map = np.zeros((h, w))
        det_map[(h * boxes[:, 1]).cpu().numpy().astype(int), (w * boxes[:, 0]).cpu().numpy().astype(int)] = 1
        det_map = ndimage.gaussian_filter(
            det_map, sigma=(5, 5), order=0
        )
        print(det_map.any())
        plt.imshow(image[f_name])
        plt.imshow(det_map[None, :].transpose(1, 2, 0), 'jet', interpolation='none', alpha=0.7)
        
        plt.show()
        plt.savefig(f_name + "_detections.jpg")
        plt.close()
    
    abs_errs.append(err)
    
abs_errs = np.array(abs_errs)
print("MAE: " + str(np.mean(abs_errs)))
print("RMSE: " + str(np.sqrt(np.mean(abs_errs**2))))
print(str(len(abs_errs)) + " images tested.")
