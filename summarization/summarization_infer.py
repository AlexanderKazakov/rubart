import sys
import os

sys.path.insert(0, os.getcwd())
from summarization.common import *
from summarization.modeling_rubart import RuBartForConditionalGeneration


SPORTSRU = True
if SPORTSRU:
    CKPT_PATH = DATA_PATH + 'ckpts/sportsru_1'
else:
    CKPT_PATH = DATA_PATH + 'ckpts/lenta_pretrained'

set_global_device('cpu')
set_seed(123)
set_batch_size(1)
set_max_len_src(256)
set_max_len_tgt(24)
set_min_len_tgt(1)
if SPORTSRU:
    set_max_len_tgt(256)
    set_min_len_tgt(64)

model_init, tokenizer = load_rubart_with_pretrained_encoder()
model = RuBartForConditionalGeneration.from_pretrained(CKPT_PATH).eval()

if SPORTSRU:
    train_loader, val_loader, test_loader = read_sportsru(tokenizer)
else:
    # train_loader, test_loader = read_dataset('lenta', tokenizer)
    train_loader, test_loader = read_dataset('ria', tokenizer)

model.eval()
for text, title in test_loader.dataset:
    if SPORTSRU:
        source_ids = encode_text_end(tokenizer, text, get_max_len_src())
    else:
        source_ids = encode_text(tokenizer, text, get_max_len_src())

    encoder_attention_mask = source_ids != tokenizer.pad_token_id
    generated_ids = model.generate(
        input_ids=source_ids,
        attention_mask=encoder_attention_mask,
        num_beams=4,
        min_length=get_min_len_tgt(),
        max_length=get_max_len_tgt(),
    )

    generated_title = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    pprint([text, len(text.split())])
    decoded_text = tokenizer.decode(source_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    pprint(decoded_text)
    print('-' * 50)
    print(title)
    print('-' * 50)
    print(generated_title)
    print('=' * 50 + '\n')




