# visda2019-semisupervised 
We release the source code of our submissions for Semi-Supervised Domain Adaptation task in VisDA-2019. Details can be referred in [Technical report](https://drive.google.com/open?id=1DvZ-QnSNoYnIkrv6jeVYzQ2S2bV4NDmy).

Pre-trained models and extracted scores are uploaded to https://drive.google.com/open?id=1yymuaZx4h8z9f5IPhUY87oBbh5ki9Lat .
## Folder structure
```
├── data
│   ├── clipart
│   ├── list
│   ├── painting
│   └── real
├── README.md
└── source_code
    ├── configs
    ├── datasets
    ├── evaluate_models.py
    ├── evaluate_pseudo_round_1.sh
    ├── evaluate_pseudo_round_2.sh
    ├── evaluate_pseudo_round_3.sh
    ├── evaluate_source_only.sh
    ├── evaluation.py
    ├── experiments
    ├── lib
    ├── losses
    ├── main.py
    ├── models
    ├── pretrained
    ├── __pycache__
    ├── run_pseudo_round_1.sh
    ├── run_pseudo_round_2.sh
    ├── run_pseudo_round_3.sh
    ├── run_source_only.sh
    ├── samplers
    ├── scripts
    ├── tools
    └── trainer.py
```

## prepare data and pretrained imagenet models
1. download images to `data` folder and unzip to `clipart`, `painting` and `real` folders.
2. merge source domain and labeled target domain samples
``` 
python tools/create_balance_data.py clipart
python tools/create_balance_data.py painting
```
3. pretrained ImageNet models are in `pretrained/checkpoints`

## Training and inference
1. train base models with source domain and labeled target domain samples. We will train 7 backbones including `Efficient B{4,5,6,7}`, `se_resnext101_32x4d`, `senet154` and `inceptionresnetv2`.
```
bash run_source_only.sh
```
You can skip this step since we provide pretrained models in `experiments/full_set_models`.

2. inference with models in step 1
```
bash evaluate_source_only.sh
```
You can skip this step since we provide results in `experiments/all_predictions/0920_noon/multi_crop_results`.

3. fusion and submission
```
python ./tools/ensemble_models_avg.py experiments/all_predictions/0920_noon clipart ./experiments/submission/v1
python ./tools/ensemble_models_avg.py experiments/all_predictions/0920_noon painting ./experiments/submission/v1
cat ./experiments/submission/v1/clipart_pred.txt ./experiments/submission/v1/painting_pred.txt > ./experiments/submission/v1/result.txt
```
This submission achieves 64.3%.

4. create pseudo labels round_1
```
python tools/create_pseudo_labels.py 1 clipart
python tools/create_pseudo_labels.py 1 painting
```
You can skip this step since we provide pre-computed pseudo labels in `data/list`.

5. train with pseudo labels round_1

```
bash run_pseudo_round_1.sh
```
You can skip this step since we provide pretrained models in `experiments/full_set_models`.

6. inference with models in step 5
```
bash evaluate_pseudo_round_1.sh
```
You can skip this step since we provide results in `experiments/all_predictions/0922_night/multi_crop_results`.

7. fusion and submission
```
python ./tools/ensemble_models_avg.py experiments/all_predictions/0922_night clipart ./experiments/submission/v2
python ./tools/ensemble_models_avg.py experiments/all_predictions/0922_night painting ./experiments/submission/v2
cat ./experiments/submission/v2/clipart_pred.txt ./experiments/submission/v2/painting_pred.txt > ./experiments/submission/v2/result.txt
```
This submission achieves 68.8%.

8. create pseudo labels round_2
```
python tools/create_pseudo_labels.py 2 clipart
python tools/create_pseudo_labels.py 2 painting
```
You can skip this step since we provide pre-computed pseudo labels in `data/list`.

9. train with pseudo labels round_2

```
bash run_pseudo_round_2.sh
```
You can skip this step since we provide pretrained models in `experiments/full_set_models`.

10. inference with models in step 9
```
bash evaluate_pseudo_round_2.sh
```
You can skip this step since we provide results in `experiments/all_predictions/0924_noon/multi_crop_results`.

11. fusion and submission
```
python ./tools/ensemble_models_avg.py experiments/all_predictions/0924_noon clipart ./experiments/submission/v3
python ./tools/ensemble_models_avg.py experiments/all_predictions/0924_noon painting ./experiments/submission/v3
cat ./experiments/submission/v3/clipart_pred.txt ./experiments/submission/v3/painting_pred.txt > ./experiments/submission/v3/result.txt
```
This submission achieves 70.5%.

12. create pseudo labels round_3

```
python tools/create_pseudo_labels.py 3 clipart
python tools/create_pseudo_labels.py 3 painting
```
You can skip this step since we provide pre-computed pseudo labels in `data/list`.

13. train with pseudo labels round_3

```
bash run_pseudo_round_3.sh
```
You can skip this step since we provide pretrained models in `experiments/full_set_models`.

14. inference with models in step 13
```
bash evaluate_pseudo_round_3.sh
```
You can skip this step since we provide results in `experiments/all_predictions/fusion_results`. We include 7 different inference settings: 1) normal inference, 2) multi crop inference, 3) high resolution inference, 4) high resolution with no cropping, 5) normal inference with model weight average, 6) high resolution inference with model weight average, 7) high resolution with no cropping and model weight average.

15. fusion and submission
```
python ./tools/ensemble_models_avg.py experiments/all_predictions/fusion_results clipart ./experiments/submission/v4
python ./tools/ensemble_models_avg.py experiments/all_predictions/fusion_results painting ./experiments/submission/v4
cat ./experiments/submission/v4/clipart_pred.txt ./experiments/submission/v4/painting_pred.txt > ./experiments/submission/v4/result.txt
```
This submission achieves 71.35%.

16. extra fusion with prototype centers. 

```
python tools/ensemble_models_final.py --domain clipart --model 0,1,2,3,4,5,6 --enable_prototype --prototye_acorss_model 1 --output ./experiments/submission/v5 --source_only_proto 1
python tools/ensemble_models_final.py --domain painting --model 0,1,2,3,4,5,6 --enable_prototype --prototye_acorss_model 1 --output ./experiments/submission/v5 --source_only_proto 1
cat ./experiments/submission/v5/clipart_pred.txt ./experiments/submission/v5/painting_pred.txt > ./experiments/submission/v5/result.txt
```
**NOTE**: To use this fusion, we need to extract features for each models for both source domain image and target domain images. The features are very large (around 260GB) so we do not include in this repo. If you run step14, these features will be extracted already. 

This submission achieves 71.41%.

## Citation
Please cite our technique report in your publications if it helps your research:

```
@article{pan2019visda,
  title={Multi-Source Domain Adaptation and Semi-Supervised Domain Adaptation with Focus on Visual Domain Adaptation Challenge 2019},
  author={Pan, Yingwei and Li, Yehao and Cai, Qi and Chen, Yang and Yao, Ting},
  booktitle={Visual Domain Adaptation Challenge},
  year={2019}
}
```
