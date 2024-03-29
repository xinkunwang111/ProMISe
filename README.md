# ProMISe: Promptable Medical Image Segmentation using SAM

The [paper](https://arxiv.org/pdf/2403.04164.pdf) has been stored in Arxiv.
Our main page in paperwithcode is [here](https://paperswithcode.com/paper/promise-promptable-medical-image-segmentation)

# 1. Introduction
With the proposal of Segment Anything Model (SAM), finetuning SAM for medical image segmentation (MIS) has become popular. However, due to the large size of the SAM model and the significant domain gap between natural and medical images, fine-tuning-based
strategies are costly with potential risk of instability, feature damage
and catastrophic forgetting. Furthermore, some methods of transferring
SAM to a domain-specific MIS through fine-tuning strategies disable the
model’s prompting capability, severely limiting its utilization scenarios.
In this paper, we propose an Auto-Prompting Module (APM), which provides SAM-based foundation model with Euclidean adaptive prompts
in the target domain. Our experiments demonstrate that such adaptive prompts significantly improve SAM’s non-fine-tuned performance
in MIS. In addition, we propose a novel non-invasive method called Incremental Pattern Shifting (IPS) to adapt SAM to specific medical domains.  Experimental results show that the IPS enables SAM to achieve
state-of-the-art or competitive performance in MIS without the need for
fine-tuning. By coupling these two methods, we propose ProMISe, an
end-to-end non-fine-tuned framework for Promptable Medical Image
Segmentation. Our experiments demonstrate that both using our methods individually or in combination achieves satisfactory performance in
low-cost pattern shifting, with all of SAM’s parameters frozen.
# 2. Framework
![image](https://github.com/xinkunwang111/ProMISe/assets/130198762/1e1ff6cf-7eb6-4ab9-a2a5-7fc28661c3a5)

# 3. Usage
## 3.1 Packages
Please see SAM.yaml.
## 3.2 Datasets
1. [cvc300(Endoscene)](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/cvc-endoscenestill/)
2. [Clinic-DB](https://polyp.grand-challenge.org/CVCClinicDB/)
3. [Colon-DB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579)
4. [ETIS-LARIBPOLYPDB](https://polyp.grand-challenge.org/ETISLarib/)
5. [Kvasir-SEG](https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset)
6. [ISIC-2018](https://challenge.isic-archive.com/data/#2018)

The input dataset csv format should follow the "combined_5_1024.csv".
1. [Rank1- ISIC2018](https://paperswithcode.com/sota/lesion-segmentation-on-isic-2018)
2. [Rank2- Colon-DB](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb)
3. [Rank2- ETIS](https://paperswithcode.com/sota/medical-image-segmentation-on-etis)
4. [Rank25- Kvasir](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg)




## 3.3 Train
Please use train_ProMISe.sh

1. `APM_resnet`: For only use APM with resnet34
2. `APM_IPS_resnet`: Use resnet34 as APM, and use IPS block
3. `IPS_GT`:Use GT points as prompt, and use IPS block

You can load the pre-trained model's checkpoint into `args.check_point_path`.

## 3.4 Evaluation
Please use eval_ProMISe.sh

1. `APM_resnet`: For only use APM with resnet34
2.  `APM_IPS_GT_resnet`: Use checkpoint trained in `APM_IPS_resnet`, but use GT to provide point prompts.
3.  `IPS_GT`:Use GT points as prompt, and use IPS block


You can load the trained model's checkpoint into `args.TEST_check_point_path`.

## 3.5 Checkpoints
1. [APM_resnet](https://drive.google.com/file/d/1bjyRUKolZ5ON-egnSnfpLNdyDQvcOmWL/view?usp=drive_link)
2. [APM_IPS_resnet](https://drive.google.com/file/d/1HSX4HgrrBreAoVDSUcOZhpN8BnKEnJO-/view?usp=drive_link)
3. [IPS_GT](https://drive.google.com/file/d/1R1eqzYkEjoynSynn8OP4maL6YjZjgW-f/view?usp=drive_link)

# 4. Notes
## 4.1  Random Seed
In training process with `IPS_GT`, the random seed is important. In order to keep the stability and generalization when evaluation, we recommend you do not set random seed in training  process. If you choose random seed, please keep the same random seed when you do the evaluation. This might help you get better performance.
## 4.2 Postprocessing
We use ` cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)` to do the postprocessing. However, the best choice of `kernel size` and `cv2.MORPH_OPEN/cv2.MORPH_CLOSE` may vary with models and datasets.

# 5. Acknowledge
We are very grateful for the endeavour and works from Meta. Their work on SAM provide the fundament for our framework.



   


