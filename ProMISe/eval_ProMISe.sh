###################################
### APM_resnet 16points ###
###################################
GPU_num='0'
net_name=SAM_vit-b_APM_resnet_train
CUR_PATH='v02' 
FREEZE_TYPE="_FZ0" ## APM_IPS_GT_cross:FZ0-69, FZ1-85, FZ2-189, FZ3-205
BATCH=8
LOSS_TYPE='BCEDiceLoss'
INITIAL_LR='1e-5'
EPOCH=200
LR_PARA='LR_200_1e-7'
point_num=16 #16
point_value='1-1-1-1-1-1-1-1-0-0-0-0-0-0-0-0' #'1-1-1-1-1-1-1-1-0-0-0-0-0-0-0-0'
###################################

########################################################
### IPS 16points ###
########################################################
GPU_num='1'
net_name=SAM_vit-b_IPS_GT_resnet_train
CUR_PATH='v02' 
FREEZE_TYPE="_FZ0" ## APM_IPS_GT_cross:FZ0-69, FZ1-85, FZ2-189, FZ3-205
BATCH=8
LOSS_TYPE='BCEDiceLoss'
INITIAL_LR='1e-5'
EPOCH=200
LR_PARA='LR_200_1e-7'
point_num=16
point_value='1-1-1-1-1-1-1-1-0-0-0-0-0-0-0-0'
########################################################

###################################
### APM_IPS_resnet 16points ###
###################################
GPU_num='2'
net_name=SAM_vit-b_APM_IPS_resnet_train
CUR_PATH='v02' 
FREEZE_TYPE="_FZ0"
BATCH=8
LOSS_TYPE='BCEDiceLoss'
INITIAL_LR='1e-5'
EPOCH=200
LR_PARA='LR_200_1e-7'
point_num=16
point_value='1-1-1-1-1-1-1-1-0-0-0-0-0-0-0-0'
###################################

IF_SPLIT_CSV=/mnt/DATA-1/SAM_GROUP/Datasets/polyp/combined_5_1024.csv
BEST_MODEL_PATH=/mnt/DATA-2/sifan2/Github/ProMISe/best_models/${CUR_PATH}/
check_point_path="/mnt/DATA-2/sifan2/Github/SAM/sam_vit_b_01ec64.pth"
LOG_PATH=/mnt/DATA-2/sifan2/Github/ProMISe/logs/${CUR_PATH}/
RESULT_DIR=/mnt/DATA-2/sifan2/Github/ProMISe/Results/${CUR_PATH}/

PROCESS_NAME=${net_name}${FREEZE_TYPE}_${LR_PARA}_${LOSS_TYPE}_${INITIAL_LR}_${EPOCH}_${point_num}
echo ${PROCESS_NAME} ${RESULT_DIR}

#### eval process ####
# TEST_IF_SPLIT_CSV=/mnt/DATA-1/SAM_GROUP/Datasets/polyp/TestDataset/split_Kvasir_1024.csv
# TEST_IF_SPLIT_CSV=/mnt/DATA-1/SAM_GROUP/Datasets/polyp/TestDataset/split_CVC-300_1024.csv 
# TEST_IF_SPLIT_CSV=/mnt/DATA-1/SAM_GROUP/Datasets/polyp/TestDataset/split_CVC-ColonDB_1024.csv
TEST_IF_SPLIT_CSV=/mnt/DATA-1/SAM_GROUP/Datasets/polyp/TestDataset/split_ETIS-LaribPolypDB_1024.csv
## TEST_IF_SPLIT_CSV=/mnt/DATA-1/SAM_GROUP/Datasets/polyp/TestDataset/split_CVC-ClinicDB_1024.csv

TEST_check_point_path="/mnt/DATA-2/sifan2/Github/ProMISe/best_models/v02/${PROCESS_NAME}/model_val_best_197_0_0.496.pt"

TEST_point_num=16
TEST_net_name=${net_name} ## for SAM_vit-b_APM_resnet_train; SAM_vit-b_APM_cross_train; SAM_vit-b_IPS_GT_resnet_train
# TEST_net_name=SAM_vit-b_APM_IPS_GT_cross_train ## for SAM_vit-b_APM_IPS_GT_cross_train only 
# TEST_net_name=SAM_vit-b_APM_IPS_GT_resnet_train ## for SAM_vit-b_APM_IPS_GT_resnet_train only 

python main.py --TYPE 'eval' --process_name ${PROCESS_NAME} --check_point_path ${TEST_check_point_path} --point_num ${TEST_point_num} --point_value ${point_value} --ava_device ${GPU_num} --label_type 'seg' --New_size 1024 --transformer_train 'train2_color' --batch_size ${BATCH} --model_name ${TEST_net_name} --pretrained True --freeze_type ${FREEZE_TYPE} --loss_type ${LOSS_TYPE} --check_num_per_epoch 1 --TOTAL_EPOCH ${EPOCH} --INITIAL_LR ${INITIAL_LR} --scheduler_type ${LR_PARA} --if_split_csv ${TEST_IF_SPLIT_CSV} --best_model_path ${BEST_MODEL_PATH} --log_path ${LOG_PATH} --SEED 128 > /mnt/DATA-2/sifan2/Github/ProMISe/eval/${PROCESS_NAME}.txt

