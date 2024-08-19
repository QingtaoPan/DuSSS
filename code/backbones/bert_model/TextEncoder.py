import torch.nn as nn
import torch
from transformers import BertTokenizer, BertConfig
from backbones.bert_model.med import BertModel


class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)


class TextEncoder_Bert(nn.Module):
    def __init__(self,
                 tokenizer: BertTokenizer = None,
                 emb_dim: int = 768,
                 output_dim: int = 128,
                 hidden_dim: int = 2048,
                 freeze_bert: bool = False):
        super(TextEncoder_Bert, self).__init__()
        self.Bio_ClinicalBERT_path = '/root/data1/journal/code/backbones/bert_model/Bio_ClinicalBERT'
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_bert = freeze_bert
        self.agg_tokens = True

        self.config = BertConfig.from_json_file('/root/data1/journal/code/backbones/bert_model/bert_config.json')
        self.tokenizer = BertTokenizer.from_pretrained(self.Bio_ClinicalBERT_path)  # text-->token
        self.model = BertModel.from_pretrained(self.Bio_ClinicalBERT_path, config=self.config, add_pooling_layer=False)

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        self.global_embed = GlobalEmbedding(self.embedding_dim, hidden_dim, self.output_dim)
        self.local_embed = LocalEmbedding(self.embedding_dim, hidden_dim, self.output_dim)

    def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
        '''
        :param embeddings: bz, 1, 112, 768
        :param caption_ids: bz, 112
        :param last_layer_attn: bz, 111
        '''
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        last_attns = []

        # loop over batch
        for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            attns = []
            attn_bank = []

            # loop over sentence
            for word_emb, word_id, attn in zip(embs, caption_id, last_attn):
                word = self.idxtoword[word_id.item()]
                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))
                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    break
                # This is because some words are divided into two words.
                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(attn)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        attns.append(sum(attn_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                        attn_bank = [attn]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])
                    attn_bank.append(attn)
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.type_as(agg_embs)
            words = words + ["[PAD]"] * padding_size
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)

        return agg_embs_batch, sentences, last_atten_pt

    def forward(self, texts, device):
        tokens = self.tokenizer(texts, return_tensors="pt", truncation=True, padding="max_length", max_length=20)
        ids, attn_mask, token_type = tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"]
        ids, attn_mask, token_type = ids.to(device), attn_mask.to(device), token_type.to(device)
        outputs = self.model(ids, attn_mask, token_type, return_dict=True, mode="text")

        last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)
        all_feat = outputs.last_hidden_state.unsqueeze(1)

        all_feat, sents, last_atten_pt = self.aggregate_tokens(all_feat, ids, last_layer_attn)

        if self.last_n_layers == 1:
            all_feat = all_feat[:, 0]

        report_feat = all_feat[:, 0].contiguous()
        word_feat = all_feat[:, 1:].contiguous()


        return report_feat, word_feat, last_atten_pt, sents


# model = TextEncoder_Bert()
#
# report_feat, word_feat, last_atten_pt, sents = model(['i love you to', 'my dog'])
# print(report_feat.shape)
# print(word_feat.shape)
# print(last_atten_pt)
# print(sents)

