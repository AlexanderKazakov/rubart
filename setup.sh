#!/usr/bin/env bash

apt-get install unzip
pip install matplotlib transformers rouge pprint tqdm sklearn

git clone https://github.com/AlexanderKazakov/rubart
cd rubart

mkdir -p data/ckpts
cd data
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12jm_uRXynQxmRrgteWXArqcqQYm5jHJK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12jm_uRXynQxmRrgteWXArqcqQYm5jHJK" -O lenta-ru-news.zip
rm cookies.txt
unzip lenta-ru-news.zip

cd ckpts
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vOMC9U3Bddgg9M-LEX12U92LNRv6h--t' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vOMC9U3Bddgg9M-LEX12U92LNRv6h--t" -O rubart_initial_weights_from_rubert.zip
rm cookies.txt
unzip rubart_initial_weights_from_rubert.zip


# run train on lenta:
# python summarization/summarization.py --device='cuda' --seed=123 --max_source_length=256 --max_target_length=24 --batch_size=32


