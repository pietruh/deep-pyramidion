# Sparsifying Transformer Models with Trainable Representation Pooling (Pietruszka et. al, 2020)

https://arxiv.org/abs/2009.05169

## Introduction

The repository contains implementation of Pyramidion model, that achieves sublinear complexity with respect to the sequence length.
We optimize the attention complexity by learning to select encoded representations for the given task and _selecting_ only the chosen ones to the next layer of the model.

The released models perform on SOTA level for long document summarization task while being __4.5x__ faster in inference than competitive models based on blockwise attention.

## Data preparation
The authors use the sentencepiece vocabs available [here](TODO: add links). 

## Training

````
DATA = /data-c/shared/disconet/old/discovery/skynet/experiments/s2s/summarize-arxiv/data/summarization-dataset-format
fairseq-train ${DATA} \
    --user-dir examples/deep_pyramidion/ \
    --task translation \
    --arch pyramidion_base \
    --dataset-impl mmap \
    --optimizer adam \
    --encoder-pooling topk \
    --enc-layers-and-token-width "{0: 2048, 1: 512}" \
    --max-source-positions 2048 \
    --max-target-positions 1024 \
    --chunk-size 512 \
    --truncate-source \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --source-lang src \
    --target-lang tgt \
    --valid-subset dev-0 \
    --max-sentences 16 \
    --update-freq 16 \
    --warmup-updates 5000 \
    --patience 220 \
    --lr 5e-4 \
    --max-update 300000 \
    --total-num-update 300000 \
    --lr-scheduler polynomial_decay \
    --weight-decay 0.1 \
    --share-all-embeddings \
    --save-dir results/ \
    --sort-back 1 \
    --flip-right 1 \
    --num-workers 4 \
    --attention-dropout 0.1 \
    --dropout 0.1
````

## Inference
````
fairseq-generate ${DATA} \
        --user-dir examples/deep_pyramidion/ \
        --task translation \
        --dataset-impl mmap \
        --optimizer adam \
        --max-source-positions 8192 \
        --truncate-source \
        --path /data-c/shared/disconet/old/discovery/skynet/experiments/s2s/summarize-arxiv/experiments/195-official-release-heavy-refactor-after-fixes/results/train/restore-194--deep-6x6-1/checkpoint${CKPT}.pt \
        --gen-subset test-A \
        --results-path ./generate_results \
        --bpe sentencepiece \
        --sentencepiece-model ${SPM_MODEL} \
        --batch-size 8 \
        --temperature 1.0 \
        --beam 16 \
        --lenpen 1.0 \
        --min-len 72 \
        --max-len-b 966 \
````