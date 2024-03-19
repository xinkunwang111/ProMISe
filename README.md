# ProMISe: Promptable Medical Image Segmentation using SAM
Our code will be released in the very near future.
#1. Introduction
With the proposal of Segment Anything Model (SAM), finetuning SAM for medical image segmentation (MIS) has become popular. However, due to the large size of the SAM model and the significant domain gap between natural and medical images, fine-tuning-based
strategies are costly with potential risk of instability, feature damage
and catastrophic forgetting. Furthermore, some methods of transferring
SAM to a domain-specific MIS through fine-tuning strategies disable the
model’s prompting capability, severely limiting its utilization scenarios.
In this paper, we propose an Auto-Prompting Module (APM), which provides SAM-based foundation model with Euclidean adaptive prompts
in the target domain. Our experiments demonstrate that such adaptive prompts significantly improve SAM’s non-fine-tuned performance
in MIS. In addition, we propose a novel non-invasive method called Incremental Pattern Shifting (IPS) to adapt SAM to specific medical domains. Experimental results show that the IPS enables SAM to achieve
state-of-the-art or competitive performance in MIS without the need for
fine-tuning. By coupling these two methods, we propose ProMISe, an
end-to-end non-fine-tuned framework for Promptable Medical Image
Segmentation. Our experiments demonstrate that both using our methods individually or in combination achieves satisfactory performance in
low-cost pattern shifting, with all of SAM’s parameters frozen.
# 2. Framework
![image](https://github.com/xinkunwang111/ProMISe/assets/130198762/1e1ff6cf-7eb6-4ab9-a2a5-7fc28661c3a5)

# 3. Usage
## 3.1 Packages
Please see requirement.txt(will be relased very soon)
## 3.2 Datasets
CVC-300/
Clinc-DB/
Clono-DB/
ETIS-LARIBPOLYPDB/
Kvasir-SEG
## 3.3 Train
You can change the train model by adding these phrase in args.net_name.
1. APM_cross: For only use APM with cross attention
2. APM_resnet: For only use APM with resnet34

3. APM_IPS_cross: Use cross attention as APM, and use IPS block
4. APM_IPS_resnet: Use resnet34 as APM, and use IPS block

5. IPS_GT:Use GT points as prompt, and use IPS block

You can load the pre-trained model's checkpoint into args.check_point_path.

## 3.4 Evaluation
You can change the evaluation model by adding these phrase in args.net_name.
1. APM_cross_test: For only use APM with cross attention
2. APM_resnet_test: For only use APM with resnet34

3. APM_IPS_cross_test: Use cross attention as APM, and use IPS block
4. APM_IPS_resnet_test: Use resnet34 as APM, and use IPS block

5. IPS_GT_test:Use GT points as prompt, and use IPS block

You can load the trained model's checkpoint into args.check_point_path.



# 4. Acknowledge
We are very grateful for the endeavour and works from Meta. Their works on SAM provide the fundament for our framework.



   


