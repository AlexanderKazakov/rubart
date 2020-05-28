import sys
import os
sys.path.insert(0, os.getcwd())
from summarization.common import *

from transformers import BartForConditionalGeneration, BartTokenizer


def summarize(txt_input, **kwargs):
    inp = encode_text(tokenizer, txt_input)
    print(inp.shape, tokenizer.decode(inp[0]), tokenizer.tokenize(txt_input))
    summary_ids = bart.generate(inp, **kwargs)
    print([decode_text(tokenizer, g) for g in summary_ids])


bart_ckpt_name = 'bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(bart_ckpt_name, do_lowercase=False)

print(tokenizer.get_vocab()['Putin'])
print(tokenizer.tokenize('Putin'))
enc_txt = encode_text(tokenizer, 'Putin')
print(enc_txt)
print(decode_text(tokenizer, enc_txt))

print(type(tokenizer))
bart = BartForConditionalGeneration.from_pretrained(bart_ckpt_name)
bart.eval()
print(type(bart))

summarize("""
Today is a sad day for us here at Telegram. We are announcing the discontinuation of our blockchain project. Below is a summary of what it was and why we had to abandon it.
""", num_beams=4, max_length=15, early_stopping=False, temperature=1)





