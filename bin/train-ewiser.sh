#!/bin/bash

MODEL='ewiser-hyper-lmms+sensembert'
CORPUS_DIR='../res/experiments/semcor+wngt'
EMBEDDINGS='../res/embeddings/sensembert+lmms.svd512.synset-centroid.vec'
EDGES='../res/edges'
EPOCHS=50

SAVEDIR=${CORPUS_DIR}/checkpoints/${MODEL}
mkdir -p $SAVEDIR

args1=(\
--arch linear_seq \
--task sequence_tagging \
--criterion weighted_cross_entropy \
--tokens-per-sample 100 \
--max-tokens 1000 \
--optimizer adam \
--min-lr 1e-7 \
--lr-scheduler fixed \
--decoder-embed-dim 512 \
--update-freq 4 \
--dropout 0.2 \
--clip-norm 1.0  \
--context-embeddings \
--context-embeddings-type bert \
--context-embeddings-bert-model roberta-large \
--context-embeddings-cache \
--only-use-targets \
--log-format tqdm \
--decoder-layers 2 \
--decoder-norm \
--decoder-last-activation \
--decoder-activation swish \
--no-epoch-checkpoints
)

echo $SAVEDIR

args2=( --decoder-output-pretrained $EMBEDDINGS --decoder-use-structured-logits --decoder-structured-logits-edgelists \
	${EDGES}/hypernyms.tsv \
#	${EDGES}/hyponyms.tsv \
	${EDGES}/derivationally.sym.tsv \
	${EDGES}/similartos.sym.tsv \
	${EDGES}/verbgroups.sym.tsv )


# BASELINE
# CUDA_VISIBLE_DEVICES=0 python train.py $CORPUS_DIR "${args1[@]}" --lr 1e-4 --save-dir $SAVEDIR-baseline --max-epoch $EPOCHS

# EWISER training stage 1
CUDA_VISIBLE_DEVICES=0 python train.py $CORPUS_DIR "${args1[@]}" "${args2[@]}" --lr 1e-4 --save-dir $SAVEDIR-fixedemb-traingraph --max-epoch $EPOCHS \
	--decoder-output-fixed --decoder-structured-logits-trainable

if [ -f $SAVEDIR-fixedemb-traingraph/unfreeze-1e-5/checkpoint_last.pt ] ; then
	args3=(--restore-file checkpoint_last.pt --decoder-structured-logits-trainable --reset-optimizer --reset-dataloader --reset-meters )
else
	mkdir -p $SAVEDIR-fixedemb-traingraph/unfreeze-1e-5
	cp $SAVEDIR-fixedemb-traingraph/checkpoint_best.pt $SAVEDIR-fixedemb-traingraph/unfreeze-1e-5/init.pt
	args3=(--restore-file $SAVEDIR-fixedemb-traingraph/unfreeze-1e-5/init.pt --decoder-structured-logits-trainable --only-load-weights --reset-optimizer --reset-dataloader --reset-meters)
fi

# EWISER training stage 2
CUDA_VISIBLE_DEVICES=0 python train.py $CORPUS_DIR "${args1[@]}" "${args2[@]}" "${args3[@]}"\
	--lr 1e-5 \
	--save-dir $SAVEDIR-fixedemb-traingraph/unfreeze-1e-5 \
	--max-epoch 20
