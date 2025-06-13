## Usage
* QaTa-COV19 Dataset - [Link](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)
* MosMedData+ Dataset - [Link)](http://medicalsegmentation.com/covid19/)
* MoNuSeG Dataset (demo dataset) - [Link](https://drive.google.com/drive/folders/1T4ldxMgGlLT3ji2yEqaPNq2Ppf4sjiuz?usp=share_link)
### 1. VLM-pretraining
Run VLM pre-training, use the following command.
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

