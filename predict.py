import os
import os.path as osp
import torch

from PIL import Image
from typing import Optional, Any, Union, List
from torch import nn
from pydantic import BaseModel

from datasets_inference import transforms as cgd_transforms
from util.slconfig import SLConfig
from util.misc import nested_tensor_from_tensor_list
from groundingdino.util.box_ops import box_cxcywh_to_xyxy

import gdown
import wget
from transformers.models.bert import BertConfig, BertModel
from transformers import AutoTokenizer


DEFAULT_CONFIDENCE = 0.23
CURRENT_DIR = osp.dirname(osp.abspath(__file__))
CHECKPOINT_DIR = osp.join(CURRENT_DIR, "checkpoints")


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        if url.startswith("https://drive.google.com"):
            gdown.download(url, path, quiet=False, fuzzy=True)
        else:
            wget.download(url, out=path)
    return path


class CountInferenceData(BaseModel):
    """
    Represents an inference result from the Count model.

    Attributes:
        label (str): The predicted label for the detected object.
        score (float): The confidence score associated with the prediction (between 0 and 1).
        bbox (list[float]): A list of four floats representing the bounding box coordinates (xmin, ymin, xmax, ymax)
                          of the detected object in the image.
    """

    label: str
    score: float
    bbox: list[float]


class CountGDCounting:
    def __init__(self, device) -> None:
        CHECKPOINT = (
            "https://drive.google.com/file/d/1JpfsZtcGLUM0j05CpTN4kCOrmrLf_vLz/view?usp=sharing",
            os.path.join(CHECKPOINT_DIR, "counting.pth"),
        )
        BERT_CHECKPOINT = (
            "bert-base-uncased",
            os.path.join(CHECKPOINT_DIR, "bert-base-uncased"),
        )
        CFG_FILE = self.config_file = osp.join(
            CURRENT_DIR,
            "config",
            "cfg_fsc147_test.py",
        )

        # download required checkpoints
        self.model_checkpoint_path = download(CHECKPOINT[0], CHECKPOINT[1])

        if not os.path.exists(BERT_CHECKPOINT[1]):
            config = BertConfig.from_pretrained(BERT_CHECKPOINT[0])
            model = BertModel.from_pretrained(
                BERT_CHECKPOINT[0], add_pooling_layer=False, config=config
            )
            tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT[0])

            config.save_pretrained(BERT_CHECKPOINT[1])
            model.save_pretrained(BERT_CHECKPOINT[1])
            tokenizer.save_pretrained(BERT_CHECKPOINT[1])

        # setup device
        self.device = device

        # build model
        print("building model ... ...")
        cfg = SLConfig.fromfile(CFG_FILE)
        model = self.build_model_main(cfg, BERT_CHECKPOINT[1])
        checkpoint = torch.load(self.model_checkpoint_path, map_location="cpu")["model"]
        model.load_state_dict(checkpoint, strict=False)
        self.model = model.eval().to(device)
        print("build model, done.")

        # build pre-processing transforms
        normalize = cgd_transforms.Compose(
            [
                cgd_transforms.ToTensor(),
                cgd_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.data_transform = cgd_transforms.Compose(
            [
                cgd_transforms.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    def build_model_main(self, cfg, text_encoder):
        # we use register to maintain models from catdet6 on.
        from models_inference.registry import MODULE_BUILD_FUNCS

        assert cfg.modelname in MODULE_BUILD_FUNCS._module_dict
        # Add required args to cfg
        cfg.device = self.device
        cfg.text_encoder_type = text_encoder
        build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
        model, _, _ = build_func(cfg)
        return model

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, Image.Image],
        text: str,
        visual_prompts: List[List[float]],
        threshold: float,
    ):
        assert text != "" or len(
            visual_prompts
        ), "Either text or prompts should be provided"
        if visual_prompts:
            assert len(visual_prompts) < 4, "Only max 3 visual prompts are supported"

        prompts = {"image": image, "points": visual_prompts}
        input_image, _ = self.data_transform(image, {"exemplars": torch.tensor([])})
        input_image = input_image.unsqueeze(0).to(self.device)
        exemplars = prompts["points"]

        input_image_exemplars, exemplars = self.data_transform(
            prompts["image"], {"exemplars": torch.tensor(exemplars)}
        )
        input_image_exemplars = input_image_exemplars.unsqueeze(0).to(self.device)
        exemplars = [exemplars["exemplars"].to(self.device)]

        model_output = self.model(
            nested_tensor_from_tensor_list(input_image),
            nested_tensor_from_tensor_list(input_image_exemplars),
            exemplars,
            [torch.tensor([0]).to(self.device) for _ in range(len(input_image))],
            captions=[text + " ."] * len(input_image),
        )

        ind_to_filter = list(range(len(model_output["token"][0].word_ids)))
        logits = model_output["pred_logits"].sigmoid()[0][:, ind_to_filter]
        boxes = model_output["pred_boxes"][0]

        # Filter out low confidence detections
        box_mask = logits.max(dim=-1).values > threshold
        logits = logits[box_mask, :].cpu().numpy()
        boxes = box_cxcywh_to_xyxy(boxes[box_mask, :])
        boxes = boxes.cpu().numpy()

        # create output structure required by va
        assert len(boxes) == len(logits)
        result = []
        labels = text.split(" .") if text.strip() != "" else ["object"]

        for i in range(len(boxes)):
            if len(labels) == 1:
                lbl = labels[0]
            else:
                lbl_token_ids = [self.model.tokenizer(x)["input_ids"] for x in labels]
                pred_token_id = model_output["token"][0].ids[int(logits[i].argmax())]
                for i, lbl_token_id in enumerate(lbl_token_ids):
                    if pred_token_id in lbl_token_id:
                        lbl = labels[i]
                        break
            result.append(
                CountInferenceData(
                    **{
                        "bbox": boxes[i].tolist(),
                        "score": float(logits[i].max()),
                        "label": lbl,
                    }
                )
            )

        out_label = "Detected instances predicted with"
        if len(text.strip()) > 0:
            out_label += " text"
            if exemplars[0].size()[0] == 1:
                out_label += " and " + str(exemplars[0].size()[0]) + " visual exemplar."
            elif exemplars[0].size()[0] > 1:
                out_label += (
                    " and " + str(exemplars[0].size()[0]) + " visual exemplars."
                )
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
        # save viz
        # output_img = overlay_bounding_boxes(np.array(image), result)
        # output_img = Image.fromarray(output_img)
        return result


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CountGDCounting(device)
    image = Image.open(
        "/home/shankar/repos/vision-agent-benchmark/data/count_people.jpg"
    )
    text = "a person"
    result = model(image, text, [])
    print(result)
