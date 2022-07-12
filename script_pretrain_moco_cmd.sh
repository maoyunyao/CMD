#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python pretrain_moco_cmd.py \
--lr 0.01 \
--batch-size 64 \
--teacher-t 0.05 \
--student-t 0.1 \
--topk 8192 \
--mlp \
--contrast-t 0.07 \
--contrast-k 16384 \
--checkpoint-path ./checkpoints/pretrain_moco_cmd/ntu60_cross_subject \
--schedule 351 \
--epochs 451 \
--pre-dataset ntu60 \
--skeleton-representation graph-based \
--protocol cross_subject


CUDA_VISIBLE_DEVICES=0 python pretrain_moco_cmd.py \
--lr 0.01 \
--batch-size 64 \
--teacher-t 0.05 \
--student-t 0.1 \
--topk 8192 \
--mlp \
--contrast-t 0.07 \
--contrast-k 16384 \
--checkpoint-path ./checkpoints/pretrain_moco_cmd/ntu60_cross_view \
--schedule 351 \
--epochs 451 \
--pre-dataset ntu60 \
--skeleton-representation graph-based \
--protocol cross_view


CUDA_VISIBLE_DEVICES=0 python pretrain_moco_cmd.py \
--lr 0.01 \
--batch-size 64 \
--teacher-t 0.05 \
--student-t 0.1 \
--topk 8192 \
--mlp \
--contrast-t 0.07 \
--contrast-k 16384 \
--checkpoint-path ./checkpoints/pretrain_moco_cmd/ntu120_cross_subject \
--schedule 351 \
--epochs 451 \
--pre-dataset ntu120 \
--skeleton-representation graph-based \
--protocol cross_subject


CUDA_VISIBLE_DEVICES=0 python pretrain_moco_cmd.py \
--lr 0.01 \
--batch-size 64 \
--teacher-t 0.05 \
--student-t 0.1 \
--topk 8192 \
--mlp \
--contrast-t 0.07 \
--contrast-k 16384 \
--checkpoint-path ./checkpoints/pretrain_moco_cmd/ntu60_cross_setup \
--schedule 351 \
--epochs 451 \
--pre-dataset ntu120 \
--skeleton-representation graph-based \
--protocol cross_setup


CUDA_VISIBLE_DEVICES=0 python pretrain_moco_cmd.py \
--lr 0.01 \
--batch-size 64 \
--teacher-t 0.05 \
--student-t 0.1 \
--topk 8192 \
--mlp \
--contrast-t 0.07 \
--contrast-k 16384 \
--checkpoint-path ./checkpoints/pretrain_moco_cmd/pku_v2_cross_subject \
--schedule 801 \
--epochs 1001 \
--pre-dataset pku_v2 \
--skeleton-representation graph-based \
--protocol cross_subject