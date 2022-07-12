# Cross-view
for((i=1;i<=5;i++)); 
do  
CUDA_VISIBLE_DEVICES=0 python action_classification_cmd_semi.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu60_cross_view/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_view_semi \
  --data-ratio 0.01 \
  --finetune-skeleton-representation graph-based >> cmd_ntu60_cview_semi_0.01.txt
done

for((i=1;i<=5;i++)); 
do  
CUDA_VISIBLE_DEVICES=0 python action_classification_cmd_semi.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu60_cross_view/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_view_semi \
  --data-ratio 0.05 \
  --finetune-skeleton-representation graph-based >> cmd_ntu60_cview_semi_0.05.txt
done

for((i=1;i<=5;i++)); 
do  
CUDA_VISIBLE_DEVICES=0 python action_classification_cmd_semi.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu60_cross_view/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_view_semi \
  --data-ratio 0.1 \
  --finetune-skeleton-representation graph-based >> cmd_ntu60_cview_semi_0.1.txt
done

for((i=1;i<=5;i++)); 
do  
CUDA_VISIBLE_DEVICES=0 python action_classification_cmd_semi.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu60_cross_view/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_view_semi \
  --data-ratio 0.2 \
  --finetune-skeleton-representation graph-based >> cmd_ntu60_cview_semi_0.2.txt
done


# Cross-subject
for((i=1;i<=5;i++)); 
do  
CUDA_VISIBLE_DEVICES=0 python action_classification_cmd_semi.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu60_cross_subject/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_subject_semi \
  --data-ratio 0.01 \
  --finetune-skeleton-representation graph-based >> cmd_ntu60_csub_semi_0.01.txt
done

for((i=1;i<=5;i++)); 
do  
CUDA_VISIBLE_DEVICES=0 python action_classification_cmd_semi.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu60_cross_subject/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_subject_semi \
  --data-ratio 0.05 \
  --finetune-skeleton-representation graph-based >> cmd_ntu60_csub_semi_0.05.txt
done

for((i=1;i<=5;i++)); 
do  
CUDA_VISIBLE_DEVICES=0 python action_classification_cmd_semi.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu60_cross_subject/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_subject_semi \
  --data-ratio 0.1 \
  --finetune-skeleton-representation graph-based >> cmd_ntu60_csub_semi_0.1.txt
done

for((i=1;i<=5;i++)); 
do  
CUDA_VISIBLE_DEVICES=0 python action_classification_cmd_semi.py \
  --lr 0.01 \
  --batch-size 64 \
  --pretrained /data/user/ACTION/CMD/checkpoints/pretrain_moco_cmd/ntu60_cross_subject/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_subject_semi \
  --data-ratio 0.2 \
  --finetune-skeleton-representation graph-based >> cmd_ntu60_csub_semi_0.2.txt
done