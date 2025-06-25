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

    >>  python single_image_inference.py --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --image_path car.jpg --output_image_name "test.jpg" --text 'car'
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
        "--image_path",
        help="file path for the image",
    )
    parser.add_argument(
      "--output_image_name",
      help="name of output image",
    )
    parser.add_argument(
        "--text",
        help="text description",
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

parser = argparse.ArgumentParser("Testing Counting Model", parents=[get_args_parser()])
args = parser.parse_args()
model, transform = build_model_and_transforms(args)

input_image, target = transform(Image.open(args.image_path), {"exemplars": torch.tensor([])})
input_image = input_image.cuda()
input_exemplar = target["exemplars"].cuda()
input_text = args.text
with torch.no_grad():
    model_output = model(
        input_image.unsqueeze(0),
        [input_exemplar],
        [torch.tensor([0]).cuda()],
        captions=[input_text + " ."],
    )
logits = model_output["pred_logits"][0].sigmoid()
boxes = model_output["pred_boxes"][0]

# Only keep boxes with confidence above threshold.
box_mask = logits.max(dim=-1).values > args.confidence_thresh
logits = logits[box_mask, :]
boxes = boxes[box_mask, :]
pred_count = boxes.shape[0]

# Plot output.
(w, h) = Image.open(args.image_path).size
det_map = np.zeros((h, w))
det_map[(h * boxes[:, 1]).cpu().numpy().astype(int), (w * boxes[:, 0]).cpu().numpy().astype(int)] = 1
det_map = ndimage.gaussian_filter(
    det_map, sigma=(5, 5), order=0
)
plt.imshow(Image.open(args.image_path))
plt.imshow(det_map[None, :].transpose(1, 2, 0), 'jet', interpolation='none', alpha=0.7)

plt.show()
plt.savefig(args.output_image_name)
plt.close()

print("Count: " + str(pred_count))
