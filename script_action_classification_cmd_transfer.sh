# Semi with data ratio = 1.0 on PKU-MMD II
CUDA_VISIBLE_DEVICES=0 python action_classification_cmd_semi.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu60_cross_subject/checkpoint_0450.pth.tar \
  --finetune-dataset pku_v2 \
  --protocol cross_subject_semi \
  --data-ratio 1.0 \
  --finetune-skeleton-representation graph-based


CUDA_VISIBLE_DEVICES=0 python action_classification_cmd_semi.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu120_cross_subject/checkpoint_0450.pth.tar \
  --finetune-dataset pku_v2 \
  --protocol cross_subject_semi \
  --data-ratio 1.0 \
  --finetune-skeleton-representation graph-based