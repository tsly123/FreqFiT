data_name=$1
encoder=$2                # mae_vitb16, mocov3_vitb16, clip_vitb16
batch_size=$3
base_lr=$4
wd_lr=$5
num_tokens=$6
adapter_ratio=$7

model_root="pretrained"
model_type="ssl-vit"
freq_filter=True
optim=sgd

data_path=""              # data folder
output_dir="results"             # result folder


if [ $data_name = "flowers" ]
then
  data_name="vtab-oxford_flowers102"
  num_class=102
elif [ $data_name = "sun397" ]
then
  data_name="vtab-sun397"
  num_class=397
elif [ $data_name = "pets" ]
then
  data_name="vtab-oxford_iiit_pet"
  num_class=37
elif [ $data_name = "caltech101" ]
then
  data_name="vtab-caltech101"
  num_class=102
elif [ $data_name = "cifar100" ]
then
  data_name="vtab-cifar(num_classes=100)"
  num_class=100
elif [ $data_name = "dtd" ]
then
  data_name="vtab-dtd"
  num_class=47
elif [ $data_name = "svhn" ]
then
  data_name="vtab-svhn"
  num_class=10
fi


echo $data_name $num_class $num_tokens
echo $encoder
echo $seed
echo $base_lr
echo $wd_lr
echo $output_dir
echo $data_path

# linear
#    --config-file configs/linear/cub.yaml \
#    MODEL.FILTER "${freq_filter}" \

# bias
#    --config-file configs/finetune/cub.yaml \
#    MODEL.TRANSFER_TYPE "tinytl-bias" \
#    MODEL.FILTER "${freq_filter}" \

# adapter
#    --config-file configs/finetune/cub.yaml \
#    MODEL.TRANSFER_TYPE "adapter" \
#    MODEL.ADAPTER.REDUCATION_FACTOR "${adapter_ratio}" \
#    MODEL.ADAPTER.FILTER "${freq_filter}" \

# vpt
#    --config-file configs/prompt/cub.yaml \
#    MODEL.PROMPT.NUM_TOKENS "${num_tokens}" \
#    MODEL.PROMPT.DEEP "True" \
#    MODEL.PROMPT.DROPOUT "0.1" \
#    MODEL.PROMPT.FILTER "${freq_filter}" \

for seed in "42" "44" "82" "100" "800"; do
    python3 train.py \
            --config-file configs/linear/cub.yaml \
            MODEL.FILTER "${freq_filter}" \
            MODEL.TYPE "${model_type}" \
            DATA.BATCH_SIZE "${batch_size}" \
            DATA.FEATURE "${encoder}" \
            DATA.NAME "${data_name}" \
            DATA.NUMBER_CLASSES "${num_class}" \
            SOLVER.BASE_LR "${base_lr}" \
            SOLVER.WEIGHT_DECAY "${wd_lr}" \
            SOLVER.OPTIMIZER "${optim}" \
            SEED ${seed} \
            MODEL.MODEL_ROOT "${model_root}" \
            DATA.DATAPATH "${data_path}" \
            OUTPUT_DIR "${output_dir}/seed${seed}"
done
