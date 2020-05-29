import random
import re
import torch
import csv
import os
import json
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BartConfig, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from rouge import Rouge
from pprint import pprint
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def set_global_device(device):
    global DEVICE
    DEVICE = device
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(0))
    print('Using', DEVICE)

def get_global_device():
    global DEVICE
    return DEVICE


def set_batch_size(batch_size):
    global BATCH_SIZE
    BATCH_SIZE = batch_size

def get_batch_size():
    global BATCH_SIZE
    return BATCH_SIZE

def set_max_len_src(max_len_src):
    global MAX_LEN_SRC
    MAX_LEN_SRC = max_len_src

def get_max_len_src():
    global MAX_LEN_SRC
    return MAX_LEN_SRC

def set_max_len_tgt(max_len_tgt):
    global MAX_LEN_TGT
    MAX_LEN_TGT = max_len_tgt

def get_max_len_tgt():
    global MAX_LEN_TGT
    return MAX_LEN_TGT

def set_min_len_tgt(min_len_tgt):
    global MIN_LEN_TGT
    MIN_LEN_TGT = min_len_tgt

def get_min_len_tgt():
    global MIN_LEN_TGT
    return MIN_LEN_TGT


def set_seed(seed):
    # note: there are another nuances for gpu and multi-gpu
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# logging.basicConfig(level=logging.INFO)

DATA_PATH = 'data/'
CKPT_DIR = 'rubart_checkpoints/'
RUBART_ENC_WEIGHTS_DIR = DATA_PATH + 'ckpts/rubart_initial_weights_from_rubert/'

BATCH_SIZE = None

# for lenta data optimal is 512 / 24, for ria data -- 1024 / 24, for sportsru -- 4000 / 200
MAX_LEN_SRC = None
MAX_LEN_TGT = None
MIN_LEN_TGT = None


def encode_text(tokenizer, texts, max_len):
    if isinstance(texts, str):
        texts = [texts]
    assert isinstance(texts, list)
    enc_texts = [tokenizer.encode(txt, return_tensors='pt', max_length=max_len).squeeze(0) for txt in texts]
    texts_batch = pad_sequence(enc_texts, batch_first=True, padding_value=tokenizer.pad_token_id)
    return texts_batch


def decode_text(tokenizer, vocab_ids):
    return tokenizer.decode(
        vocab_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)


def read_data_lenta(path=DATA_PATH + 'lenta-ru-news.csv', clip_length=True):
    texts, titles = [], []
    with open(path, newline='', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for counter, (url, title, text, topic, tags, date) in enumerate(reader):
            # if counter == 10 * get_batch_size():  # COLAB
            #     break
            if clip_length:
                text = ' '.join(text.split()[:get_max_len_src()])
                title = ' '.join(title.split()[:get_max_len_tgt()])
            texts.append(text)
            titles.append(title)

    return texts, titles


def read_data_ria(path=DATA_PATH + 'processed-ria.json', clip_length=True):
    texts, titles = [], []
    with open(path, encoding='utf8') as f:
        for counter, line in enumerate(f):
            # if counter > 10 * get_batch_size():  # COLAB
            #     break

            data = json.loads(line)
            text = re.sub('<[^>]+>', '', data['text']).replace('\n', ' ').strip()
            title = re.sub('<[^>]+>', '', data['title']).replace('\n', ' ').strip()
            if clip_length:
                text = ' '.join(text.split()[:get_max_len_src()])
                title = ' '.join(title.split()[:get_max_len_tgt()])

            texts.append(text)
            titles.append(title)

    return texts, titles


def read_data_sportsru(path=DATA_PATH + 'sportsru/', clip_length=True):
    data = {
        'train': {'src': [], 'tgt': []},
        'val': {'src': [], 'tgt': []},
        'test': {'src': [], 'tgt': []},
    }
    with open(path + 'train_src.broad.txt', encoding='utf8') as f:
        for line in f:
            data['train']['src'].append(line)
    with open(path + 'train_tgt.news.txt', encoding='utf8') as f:
        for line in f:
            data['train']['tgt'].append(line)
    with open(path + 'valid_src.broad.txt', encoding='utf8') as f:
        for line in f:
            data['val']['src'].append(line)
    with open(path + 'valid_tgt.news.txt', encoding='utf8') as f:
        for line in f:
            data['val']['tgt'].append(line)
    with open(path + 'test_src.broad.txt', encoding='utf8') as f:
        for line in f:
            data['test']['src'].append(line)
    with open(path + 'test_tgt.news.txt', encoding='utf8') as f:
        for line in f:
            data['test']['tgt'].append(line)

    if clip_length:
        for k, d in data.items():
            d['src'] = [' '.join(text.split()[-get_max_len_src():]) for text in d['src']]
            d['tgt'] = [' '.join(text.split()[-get_max_len_tgt():]) for text in d['tgt']]

    # COLAB
    # for k, d in data.items():
    #     d['src'] = d['src'][:10 * get_batch_size()]
    #     d['tgt'] = d['tgt'][:10 * get_batch_size()]

    return data


def clear_or_create_directory(dir_name):
    """ ignoring all possible errors """
    shutil.rmtree(dir_name, ignore_errors=True)
    cntr = 0
    while True:
        try:
            os.makedirs(dir_name, exist_ok=True)
            return
        except OSError:
            if cntr < 10:
                # some windows bug?
                cntr += 1
                from time import sleep
                sleep(0.1 * cntr)
            else:
                raise


class SummarizationDataset(Dataset):
    def __init__(self, texts, titles):
        self.texts = texts
        self.titles = titles

    def __getitem__(self, item):
        return self.texts[item], self.titles[item]

    def __len__(self):
        return len(self.texts)


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_len_src = get_max_len_src()
        self.max_len_tgt = get_max_len_tgt()

    def __call__(self, batch):
        return (
            encode_text(self.tokenizer, [txt for txt, title in batch], self.max_len_src),
            encode_text(self.tokenizer, [title for txt, title in batch], self.max_len_tgt)
        )


def encode_text_end(tokenizer, texts, max_len):
    if isinstance(texts, str):
        texts = [texts]
    assert isinstance(texts, list)
    enc_texts = []
    for txt in texts:
        enc = tokenizer.encode(txt, return_tensors='pt').squeeze(0)
        enc = torch.cat([torch.tensor([tokenizer.convert_tokens_to_ids('[CLS]')]).long(), enc[-max_len + 1:]])
        enc_texts.append(enc)

    texts_batch = pad_sequence(enc_texts, batch_first=True, padding_value=tokenizer.pad_token_id)
    return texts_batch


class CollateFnSportsru:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_len_src = get_max_len_src()
        self.max_len_tgt = get_max_len_tgt()

    def __call__(self, batch):
        return (
            encode_text_end(self.tokenizer, [txt for txt, title in batch], self.max_len_src),
            encode_text(self.tokenizer, [title for txt, title in batch], self.max_len_tgt)
        )


def read_dataset(name, tokenizer):
    if name == 'lenta':
        all_texts, all_titles = read_data_lenta()
    else:
        assert name == 'ria'
        all_texts, all_titles = read_data_ria()

    train_texts, val_texts, train_titles, val_titles = \
        train_test_split(all_texts, all_titles, test_size=0.1, shuffle=True)
    train_dataset = SummarizationDataset(train_texts, train_titles)
    val_dataset = SummarizationDataset(val_texts, val_titles)
    collate_fn = CollateFn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=get_batch_size(), shuffle=True,
                              collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=get_batch_size(), shuffle=False,
                            collate_fn=collate_fn, num_workers=8)
    return train_loader, val_loader


def read_sportsru(tokenizer):
    data = read_data_sportsru()
    train_dataset = SummarizationDataset(data['train']['src'], data['train']['tgt'])
    val_dataset = SummarizationDataset(data['val']['src'], data['val']['tgt'])
    test_dataset = SummarizationDataset(data['test']['src'], data['test']['tgt'])
    collate_fn = CollateFnSportsru(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=get_batch_size(), shuffle=True,
                              collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=get_batch_size(), shuffle=False,
                            collate_fn=collate_fn, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=get_batch_size(), shuffle=False,
                             collate_fn=collate_fn, num_workers=8)
    return train_loader, val_loader, test_loader


def load_rubart_with_pretrained_encoder():
    from summarization.modeling_rubart import RuBartForConditionalGeneration

    tokenizer = BertTokenizer.from_pretrained(RUBART_ENC_WEIGHTS_DIR, do_lower_case=False)  # do_lower_case=False is crucial
    config = BartConfig.from_pretrained(RUBART_ENC_WEIGHTS_DIR)
    config.task_specific_params = None
    config.min_length, config.max_length = get_min_len_tgt(), get_max_len_tgt()
    print(config)

    model = RuBartForConditionalGeneration(config)
    model.model.encoder.load_state_dict(torch.load(RUBART_ENC_WEIGHTS_DIR + 'encoder_state_dict.pth'))
    # embeddings sharing
    model.model.decoder.embed_positions.weight = model.model.encoder.embed_positions.weight
    model.model.decoder.token_type_embeddings.weight = model.model.encoder.token_type_embeddings.weight
    model.model.decoder.layernorm_embedding.weight = model.model.encoder.layernorm_embedding.weight
    model.model.decoder.layernorm_embedding.bias = model.model.encoder.layernorm_embedding.bias
    assert (model.model.shared.weight == model.model.encoder.embed_tokens.weight).all()
    assert (model.model.shared.weight == model.model.decoder.embed_tokens.weight).all()
    assert (model.model.encoder.embed_positions.weight == model.model.decoder.embed_positions.weight).all()
    assert (model.model.encoder.token_type_embeddings.weight == model.model.decoder.token_type_embeddings.weight).all()
    assert (model.model.encoder.layernorm_embedding.weight == model.model.decoder.layernorm_embedding.weight).all()
    assert (model.model.encoder.layernorm_embedding.bias == model.model.decoder.layernorm_embedding.bias).all()

    # the only not pretrained parameters are decoder.layers
    return model, tokenizer


if __name__ == '__main__':
    # texts_lenta, titles_lenta = read_data_lenta(clip_length=False)
    # texts_ria, titles_ria = read_data_ria(clip_length=False)
    data_sportsru = read_data_sportsru(clip_length=False)
    tokenizer = BertTokenizer.from_pretrained(RUBART_ENC_WEIGHTS_DIR, do_lower_case=False)  # do_lower_case=False is crucial

    def explore(strings):
        enc, spl = [], []
        for t in random.sample(strings, 2000):
            enc.append(encode_text(tokenizer, t, 100000).squeeze())
            spl.append(t.split())

        len_enc = [len(e) for e in enc]
        len_spl = [len(s) for s in spl]
        print(f'enc_len / split_len = {np.median([len(e) / len(s) for e, s in zip(enc, spl)])}')
        print(f'number of samples = {len(strings)}')
        print(f'estimated total number of words = {int(np.median(len_spl) * len(strings))}')
        plt.clf()
        plt.hist(len_enc, bins=100)
        plt.hist(len_spl, bins=100)
        plt.show()

    # explore(texts_lenta)
    # explore(titles_lenta)
    # explore(texts_ria)
    # explore(titles_ria)
    explore(data_sportsru['train']['src'])
    explore(data_sportsru['train']['tgt'])






