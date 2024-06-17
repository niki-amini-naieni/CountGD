from transformers import BertConfig, BertModel
from transformers import AutoTokenizer

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config.save_pretrained("./checkpoints/bert-base-uncased")
model.save_pretrained("./checkpoints/bert-base-uncased")
tokenizer.save_pretrained("./checkpoints/bert-base-uncased")
