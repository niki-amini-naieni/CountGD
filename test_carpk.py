import random
import torch
import torchvision.transforms.functional as F
import numpy as np
import argparse
import hub
from util.slconfig import SLConfig, DictAction
import datasets_inference.transforms as T
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

# MODEL:
def get_args_parser():
    parser = argparse.ArgumentParser("Testing on CARPK", add_help=False)
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
        "--confidence_thresh",
        help="confidence threshold for model",
        default=0.25,
        type=float,
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

    parser.add_argument("--use_exemplars", action="store_true", help="use same exemplars as CounTR at test time")

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

    # visualization parameters
    parser.add_argument(
        "--save_every",
        default=1e3,
        type=int,
        help="save a plot with predicted detections every [args.save_every] samples",
    )

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
    from models_inference.registry import MODULE_BUILD_FUNCS

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

dataset_test = hub.load("hub://activeloop/carpk-test")
print(dataset_test)

data_loader_test = dataset_test.pytorch(
    num_workers=args.num_workers, batch_size=1, shuffle=False
)

abs_errs = []

for data_iter_step, data in enumerate(data_loader_test):
    # Obtain exemplars used in CounTR for CARPK (https://github.com/Verg-Avesta/CounTR/blob/2806f4559436f257585690d906598fa17820f662/FSC_test_CARPK.py#L158C1-L169C80)
    exemplars = []

    if args.use_exemplars:
      for i in range(2):
        if i == 0:
            idx = random.randint(0, int(data['boxes'].shape[1] / 2))
        else:
            idx = random.randint(int(data['boxes'].shape[1] / 2) - 1, data['boxes'].shape[1] - 1)

        box = data['boxes'][0][i]
        box2 = [int(k) for k in box]
        x1, y1, x2, y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
        exemplars.append([x1, y1, x2, y2])
    
    input_image_pil = F.to_pil_image(torch.permute(data["images"].squeeze(), (2, 0, 1)))
    input_image, target = transform(input_image_pil, {"exemplars": torch.tensor(exemplars)})
    gt_count = data["labels"].shape[1]
    input_image = input_image.cuda()
    input_exemplars = target["exemplars"].cuda()
    with torch.no_grad():
        print("exemplars.shape: " + str(input_exemplars.shape))
        model_output = model(
            input_image.unsqueeze(0),
            [input_exemplars],
            [torch.tensor([0]).cuda()],
            captions=["car ."],
        )
    logits = model_output["pred_logits"][0].sigmoid()
    boxes = model_output["pred_boxes"][0]
    # Only keep boxes that meet confidence threshold.
    box_mask = logits.max(dim=-1).values > args.confidence_thresh
    logits = logits[box_mask, :]
    boxes = boxes[box_mask, :]

    pred_count = boxes.shape[0]

    print("Pred. Count: " + str(pred_count) + ", GT Count: " + str(gt_count))
    err = abs(pred_count - gt_count)
    abs_errs.append(err)

    if (data_iter_step % args.save_every) == 0 or (args.save_results and (err == 0)):
        (w, h) = input_image_pil.size
        det_map = np.zeros((h, w))
        det_map[(h * boxes[:, 1]).cpu().numpy().astype(int), (w * boxes[:, 0]).cpu().numpy().astype(int)] = 1
        det_map = ndimage.gaussian_filter(
            det_map, sigma=(5, 5), order=0
        )
        print(det_map.any())
        plt.imshow(input_image_pil)
        plt.imshow(det_map[None, :].transpose(1, 2, 0), 'jet', interpolation='none', alpha=0.7)
        
        plt.show()
        plt.savefig(str(data_iter_step) + "_carpk_detections.jpg")
        plt.close()

abs_errs = np.array(abs_errs)
print("Number of Images Tested: " + str(len(abs_errs)))
print("MAE: " + str(np.mean(abs_errs)))
print("RMSE: " + str(np.sqrt(np.mean(abs_errs**2))))
