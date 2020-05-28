import sys
import os
sys.path.insert(0, os.getcwd())
from summarization.common import *


def collate_fn_summarization(batch):
    return (
        encode_text(tokenizer, [txt for txt, title in batch], get_max_len_src()),
        encode_text(tokenizer, [title for txt, title in batch], get_max_len_tgt())
    )


def prepare_inputs(source_ids, target_ids):
    encoder_attention_mask = source_ids != tokenizer.pad_token_id
    decoder_input_ids = target_ids[:, :-1].contiguous().clone()
    decoder_attention_mask = decoder_input_ids != tokenizer.pad_token_id
    lm_labels = target_ids[:, 1:].contiguous().clone()
    lm_labels[target_ids[:, 1:] == tokenizer.pad_token_id] = -100  # nn.CrossEntropyLoss ignore label
    return encoder_attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels


def calc_loss(source_ids, target_ids, is_train):
    source_ids, target_ids = source_ids.to(get_global_device()), target_ids.to(get_global_device())
    encoder_attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels = prepare_inputs(
        source_ids, target_ids
    )
    model.train(is_train)
    output = model(
        input_ids=source_ids,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        lm_labels=lm_labels,
        generation_mode=False
    )
    return output[0]


def train_epoch():
    losses = []
    loader = tqdm(train_loader)
    batch_counter = 0
    total_num_batches = len(train_loader)

    for source_ids, target_ids in loader:
        batch_counter += 1
        if batch_counter > total_num_batches / 5:
            loader.close()
            break

        optimizer.zero_grad()
        loss = calc_loss(source_ids, target_ids, is_train=True)
        loss.backward()
        optimizer.step()
        loss_item = loss.detach().cpu().item()
        loader.set_description(f'{loss_item:.3f}')
        losses.append(loss_item)

    return np.mean(losses)


@torch.no_grad()
def val_epoch(loader):
    losses = []
    for source_ids, target_ids in tqdm(loader):
        loss = calc_loss(source_ids, target_ids, is_train=False)
        losses.append(loss.detach().cpu().item())
    return np.mean(losses)


@torch.no_grad()
def test_generation(loader):
    total_rouge = rouge.get_scores(['a'], ['b'], avg=True)  # zero-init
    assert all(v == 0 for m, d in total_rouge.items() for s, v in d.items())
    total_num_samples = 0
    model.eval()
    save_pred, save_tgt, save_inp = [], [], []
    for source_ids, target_ids in tqdm(loader):
        source_ids, target_ids = source_ids.to(get_global_device()), target_ids.to(get_global_device())
        encoder_attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels = prepare_inputs(
            source_ids, target_ids
        )
        generated_ids = model.generate(
            input_ids=source_ids,
            attention_mask=encoder_attention_mask,
            num_beams=1
        )
        predictions = [
            tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        targets = [
            tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for t in target_ids
        ]
        RET_TXT_NUM = 10
        if len(save_pred) < RET_TXT_NUM:
            save_pred.extend(predictions)
            save_tgt.extend(targets)
            save_inp.extend([
                tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for t in source_ids
            ])

        total_num_samples += len(predictions)
        hyp_sents = [[s for s in hyp.split('.') if len(s) > 0] for hyp in predictions]
        if any(len(sents) > 0 for sents in hyp_sents):  # TODO report bugs to this rouge library
            batch_rouges = rouge.get_scores(predictions, targets, avg=False, ignore_empty=True)
            for m in total_rouge:
                for d in total_rouge[m]:
                    for sample_rouge in batch_rouges:
                        prev_val = total_rouge[m][d]
                        total_rouge[m][d] += sample_rouge[m][d]
                        assert prev_val < total_rouge[m][d] or sample_rouge[m][d] == 0  # check overflow

    for m in total_rouge:
        for d in total_rouge[m]:
            total_rouge[m][d] /= total_num_samples

    return total_rouge, save_pred, save_tgt, save_inp


def fit():
    last_ckpt_dir = CKPT_DIR + 'last_ckpt/'
    best_ckpt_dir = CKPT_DIR + 'best_ckpt/'
    clear_or_create_directory(CKPT_DIR)
    model.to(get_global_device())
    all_epoch_statistics = []
    min_val_loss = np.inf
    for epoch in range(1000):
        print(f'Epoch {epoch}')
        train_loss = train_epoch()
        print(f'Train loss: {train_loss}')
        clear_or_create_directory(last_ckpt_dir)
        model.save_pretrained(last_ckpt_dir)
        statistics = {'epoch': epoch, 'train_loss': train_loss}

        if epoch % 2 == 0:
            val_loss = val_epoch(val_loader)
            print(f'Validation loss: {val_loss}')
            statistics['val_loss'] = val_loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                clear_or_create_directory(best_ckpt_dir)
                model.save_pretrained(best_ckpt_dir)

        if epoch % 5 == 0 and epoch != 0:
            total_rouge, predictions, targets, inputs = test_generation(val_loader)
            statistics['total_rouge'], statistics['predictions'], statistics['targets'], statistics['inputs'] = \
                total_rouge, predictions, targets, inputs
            pprint(total_rouge)
            for hyp, ref, inp in zip(predictions, targets, inputs):
                print(ref)
                print(hyp)
                print('-' * 50)
                print(inp)
                print('=' * 50 + '\n')

        all_epoch_statistics.append(statistics)
        with open(CKPT_DIR + 'statistics.json', 'w') as f:
            json.dump(all_epoch_statistics, f, indent=4, sort_keys=True)

    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_source_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=24,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--seed",
        default=123,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="cuda or cpu",
    )

    args = parser.parse_args()
    set_global_device(args.device)
    set_seed(args.seed)
    set_batch_size(args.batch_size)
    set_max_len_src(args.max_source_length)
    set_max_len_tgt(args.max_target_length)

    print('get_global_device:', get_global_device())
    print('get_batch_size:', get_batch_size())
    print('get_max_len_src:', get_max_len_src())
    print('get_max_len_tgt:', get_max_len_tgt())

    all_texts, all_titles = read_data_lenta()
    train_texts, val_texts, train_titles, val_titles = train_test_split(all_texts, all_titles, test_size=0.1, shuffle=True)
    train_dataset = SummarizationDataset(train_texts, train_titles)
    val_dataset = SummarizationDataset(val_texts, val_titles)
    train_loader = DataLoader(train_dataset, batch_size=get_batch_size(), shuffle=True, collate_fn=collate_fn_summarization)
    val_loader = DataLoader(val_dataset, batch_size=get_batch_size(), shuffle=False, collate_fn=collate_fn_summarization)

    model, tokenizer = load_rubart_with_pretrained_encoder()
    # fine_tune: learning_rate = 3e-5, batch_size = 4, get_linear_schedule_with_warmup
    optimizer = AdamW(model.model.decoder.layers.parameters(), lr=1e-3)
    # TODO scheduler, weight decay with param groups, get_linear_schedule_with_warmup
    rouge = Rouge()
    fit()



