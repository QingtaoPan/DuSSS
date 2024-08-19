import torch
from transformers import BertTokenizer, BertModel
import numpy as np


def text_encode(text):
    bert_path = './bert_model'
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert = BertModel.from_pretrained(bert_path, return_dict=True, add_pooling_layer=False)
    inputs = tokenizer(text, padding='longest', max_length=10, return_tensors="pt")  # "pt"表示"pytorch"
    outputs = bert(**inputs)
    out = outputs.last_hidden_state
    if out.shape[1] > 10:
        out = out[:, :10, :]
    return out  # [b, 10, 768]


# z = []
# a = text_encode('ab v')
# z.append(a)
# b = text_encode('ab v c')
# z.append(b)
# z = torch.cat(z)
# print(z, z.shape)




