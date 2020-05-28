import sys
import os
sys.path.insert(0, os.getcwd())
from summarization.common import *


CKPT_PATH = DATA_PATH + 'ckpts/rubart_4_epochs_on_lenta'
model_init, tokenizer = load_rubart_with_pretrained_encoder()
model = RuBartForConditionalGeneration.from_pretrained(CKPT_PATH)

new_params = list(model.model.encoder.parameters()) + list(model.model.decoder.embed_tokens.parameters())
init_params = list(model_init.model.encoder.parameters()) + list(model_init.model.decoder.embed_tokens.parameters())
assert len(new_params) == len(init_params)
for new, init in zip(new_params, init_params):
    assert (new == init).all()

texts, titles = read_data_lenta(clip_length=False)
model.eval()
for text, title in zip(texts, titles):
    source_ids = encode_text(tokenizer, text, 256)
    encoder_attention_mask = source_ids != tokenizer.pad_token_id
    generated_ids = model.generate(
        input_ids=source_ids,
        attention_mask=encoder_attention_mask,
        num_beams=4
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




