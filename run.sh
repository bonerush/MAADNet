# 数据路径配置
datapath=/root/dataset/mvtec2d

# 数据集列表
datasets=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')
#datasets=('pill' 'zipper' 'toothbrush')
 
# 生成数据集标志
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

echo "Running Multi-Aspect Anomaly Detector (MAADNet) with Fourier Transform, Attention, and Simplified MoE..."

python3 main.py \
--gpu 0 \
--seed 0 \
--log_group maadnet_mvtec \
--log_project MVTecAD_MAADNet_Results \
--results_path results \
--run_name maadnet_fourier_attention_moe \
--save_segmentation_images \
--test \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 40 \
--gan_epochs 4 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin 0.5 \
--pre_proj 1 \
--noise_std 0.015 \
--use_enhanced_projection \
--use_fourier \
--use_attention \
--use_simplified_moe \
--fourier_dim 768 \
--fusion_type attention \
--dsc_dropout 0.1 \
--dsc_learnable_margin \
--fourier_use_phase \
--attention_reduction_ratio 16 \
--moe_num_experts 6 \
--moe_top_k 3 \
--moe_temperature 0.7 \
--moe_expert_dropout 0.1 \
dataset \
--batch_size 8 \
--resize 329 \
--imagesize 288 "${dataset_flags[@]}" mvtec $datapath