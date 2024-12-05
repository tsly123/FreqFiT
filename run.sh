data_name=$1
encoder=$2                # sup_vitb16_imagenet21k, mae_vitb16, mocov3_vitb16, clip_vitb16
batch_size=$3
base_lr=$4
wd_lr=$5
num_tokens=$6
adapter_ratio=$7
method=$8       # None, freqfit, ssf (scale-shift)

model_root="pretrained"
model_type="vit"
optim=sgd

data_path=""              # data folder
output_dir="results"      # result folder


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
#
elif [ $data_name = "dmlab" ]
then
  data_name="vtab-dmlab"
  num_class=6
elif [ $data_name = "clevr-distance" ]
then
  data_name='vtab-clevr(task="closest_object_distance")'
  num_class=6
elif [ $data_name = "clevr-count" ]
then
  data_name='vtab-clevr(task="count_all")'
  num_class=8
elif [ $data_name = "dsprites-orientation" ]
then
  data_name='vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)'
  num_class=16
elif [ $data_name = "dsprites-location" ]
then
  data_name='vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)'
  num_class=16
elif [ $data_name = "eurosat" ]
then
  data_name="vtab-eurosat"
  num_class=10
elif [ $data_name = "resisc45" ]
then
  data_name="vtab-resisc45"
  num_class=45
elif [ $data_name = "smallnorb-azimuth" ]
then
  data_name='vtab-smallnorb(predicted_attribute="label_azimuth")'
  num_class=18
elif [ $data_name = "smallnorb-elevation" ]
then
  data_name='vtab-smallnorb(predicted_attribute="label_elevation")'
  num_class=9
elif [ $data_name = "camelyon" ]
then
  data_name="vtab-patch_camelyon"
  num_class=2
elif [ $data_name = "kitti" ]
then
  data_name='vtab-kitti(task="closest_vehicle_distance")'
  num_class=4
elif [ $data_name = "retino" ]
then
  data_path=~/datasets
  data_name='vtab-diabetic_retinopathy(config="btgraham-300")'
  num_class=5
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

# bias
#    --config-file configs/finetune/cub.yaml \
#    MODEL.TRANSFER_TYPE "tinytl-bias" \

# adapter
#    --config-file configs/finetune/cub.yaml \
#    MODEL.TRANSFER_TYPE "adapter" \
#    MODEL.ADAPTER.REDUCATION_FACTOR "${adapter_ratio}" \

# vpt
#    --config-file configs/prompt/cub.yaml \
#    MODEL.PROMPT.NUM_TOKENS "${num_tokens}" \
#    MODEL.PROMPT.DEEP "True" \
#    MODEL.PROMPT.DROPOUT "0.1" \

# lora
#    --config-file configs/finetune/cub.yaml \
#    MODEL.TRANSFER_TYPE "lora" \
#    MODEL.LORA.RANK = 8
#    MODEL.LORA.ALPHA = 8

# boft
#    --config-file configs/finetune/cub.yaml \
#    MODEL.TRANSFER_TYPE "boft" \
#    MODEL.BOFT.BLOCK_SIZE "4" \
#    MODEL.BOFT.N_FACTOR "2" \

# vera
#    --config-file configs/finetune/cub.yaml \
#    MODEL.TRANSFER_TYPE "vera" \
#    MODEL.VERA.R "256" \

# fourierft
#    --config-file configs/finetune/cub.yaml \
#    MODEL.TRANSFER_TYPE "fft" \
#    MODEL.FFT.FREQ "3000" \
#    MODEL.FFT.SCALE "300" \

for seed in "42" "44" "82" "100" "800"; do
    python3 train.py \
            --config-file configs/finetune/cub.yaml \
            MODEL.TRANSFER_TYPE "lora" \
            MODEL.LORA.RANK "8" \
            MODEL.LORA.ALPHA "8" \
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
            OUTPUT_DIR "${output_dir}/seed${seed}" \
            FREQFIT "${method}" \
            MODEL.SAVE_CKPT "False" \
            DATA.NO_TEST "True" \
            SOLVER.SEARCH_EPOCH "3"
done
