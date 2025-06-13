## Usage
* QaTa-COV19 Dataset - [Link](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)
* MosMedData+ Dataset - [Link)](http://medicalsegmentation.com/covid19/)
* MoNuSeG Dataset (demo dataset) - [Link](https://drive.google.com/drive/folders/1T4ldxMgGlLT3ji2yEqaPNq2Ppf4sjiuz?usp=share_link)
### 1. VLM-pretraining
- Item Prepare image encoder and text encoder.
download the image encoder at: [image encoder](https://huggingface.co/pqt33/image_encoder). Put it to backbones/image_encoder
```
backbones/image_encoder/
    deit_base_patch16_224-b5f2ef4d.pth
    ImageEncoder.py
    vits.py
```
download the text encoder at:. unzip the Bio_ClinicalBERT.zip. Put it to backbones/bert_model
```
backbones/bert_model/
    Bio_ClinicalBERT
        config.py
        flax_model.msgpack
        ...
    bert_config.py
    med.py
    TextEncoder.py
```

- Item Run VLM pre-training, use the following command.
```
cd vlm_pretrain
python train_model.py
```
### 2. SSMIS
Run SSMIS, use the following command.
```
cd segmentation_train
python train_SSL.py
```

