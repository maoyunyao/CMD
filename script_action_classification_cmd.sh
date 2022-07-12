CUDA_VISIBLE_DEVICES=0 python action_classification_cmd.py \
  --lr 0.1 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu60_cross_subject/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 --protocol cross_subject --finetune-skeleton-representation graph-based

CUDA_VISIBLE_DEVICES=0 python action_classification_cmd.py \
  --lr 0.1 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu60_cross_view/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 --protocol cross_view --finetune-skeleton-representation graph-based

CUDA_VISIBLE_DEVICES=0 python action_classification_cmd.py \
  --lr 0.1 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu120_cross_subject/checkpoint_0450.pth.tar \
  --finetune-dataset ntu120 --protocol cross_subject --finetune-skeleton-representation graph-based

CUDA_VISIBLE_DEVICES=0 python action_classification_cmd.py \
  --lr 0.1 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu120_cross_setup/checkpoint_0450.pth.tar \
  --finetune-dataset ntu120 --protocol cross_setup --finetune-skeleton-representation graph-based
