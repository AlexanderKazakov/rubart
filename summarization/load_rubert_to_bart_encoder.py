import sys
import os
sys.path.insert(0, os.getcwd())
from summarization.common import *

from transformers import BertModel, BertTokenizer, BertConfig, BartConfig


rubert_ckpt_name = 'DeepPavlov/rubert-base-cased'
tokenizer = BertTokenizer.from_pretrained(rubert_ckpt_name, do_lower_case=False)  # do_lower_case=False is crucial
assert tokenizer.padding_side == 'right'
test_text_sample = 'Ай да Пушкин! синхрофазотрон'
assert tokenizer.get_vocab().get('Пушкин') is not None
assert tokenizer.tokenize(test_text_sample) == ['Ай', 'да', 'Пушкин', '!', 'синх', '##роф', '##аз', '##отрон']
enc_txt = encode_text(tokenizer, test_text_sample, max_len=32)
assert decode_text(tokenizer, enc_txt) == test_text_sample

config = BartConfig.from_pretrained('bart-large-cnn')
rubert_config = BertConfig.from_pretrained(rubert_ckpt_name)
config.model_type = 'rubart'
config.task_specific_params = None
config.vocab_size = rubert_config.vocab_size
config.pad_token_id = rubert_config.pad_token_id
config.bos_token_id = tokenizer.convert_tokens_to_ids('[CLS]')
config.eos_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
config.prefix = None
config.decoder_start_token_id = config.bos_token_id
config.max_position_embeddings = rubert_config.max_position_embeddings

# TODO choose CLS/<S>
print(tokenizer.convert_ids_to_tokens([100, 101, 102, 103, 104, 105, 106, 107]))

assert 'gelu' == config.activation_function == rubert_config.hidden_act
assert 12 == config.num_hidden_layers == config.encoder_layers == config.decoder_layers == rubert_config.num_hidden_layers
config.d_model = rubert_config.hidden_size
config.decoder_attention_heads = config.encoder_attention_heads = rubert_config.num_attention_heads
config.decoder_ffn_dim = config.encoder_ffn_dim = rubert_config.intermediate_size
config.layer_norm_eps = rubert_config.layer_norm_eps
config.attention_dropout = rubert_config.attention_probs_dropout_prob
config.dropout = rubert_config.hidden_dropout_prob
assert config.dropout == rubert_config.attention_probs_dropout_prob == rubert_config.hidden_dropout_prob == config.attention_dropout
assert config.activation_dropout == 0

model = RuBartForConditionalGeneration(config).eval()
rubert = BertModel.from_pretrained(rubert_ckpt_name).eval()
with torch.no_grad():
    # embeddings
    model.model.encoder.embed_tokens.weight = rubert.embeddings.word_embeddings.weight
    model.model.encoder.embed_positions.weight[1:, :] = rubert.embeddings.position_embeddings.weight
    model.model.encoder.token_type_embeddings.weight = rubert.embeddings.token_type_embeddings.weight
    model.model.encoder.layernorm_embedding.weight = rubert.embeddings.LayerNorm.weight
    model.model.encoder.layernorm_embedding.bias = rubert.embeddings.LayerNorm.bias
    # encoder layers
    layers_model = model.model.encoder.layers
    layers_rubert = rubert.encoder.layer
    assert 12 == len(layers_model) == len(layers_rubert)
    for layer_model, layer_rubert in zip(layers_model, layers_rubert):
        layer_model.self_attn.k_proj.weight = layer_rubert.attention.self.key.weight
        layer_model.self_attn.k_proj.bias = layer_rubert.attention.self.key.bias
        layer_model.self_attn.v_proj.weight = layer_rubert.attention.self.value.weight
        layer_model.self_attn.v_proj.bias = layer_rubert.attention.self.value.bias
        layer_model.self_attn.q_proj.weight = layer_rubert.attention.self.query.weight
        layer_model.self_attn.q_proj.bias = layer_rubert.attention.self.query.bias
        layer_model.self_attn.out_proj.weight = layer_rubert.attention.output.dense.weight
        layer_model.self_attn.out_proj.bias = layer_rubert.attention.output.dense.bias
        layer_model.self_attn_layer_norm.weight = layer_rubert.attention.output.LayerNorm.weight
        layer_model.self_attn_layer_norm.bias = layer_rubert.attention.output.LayerNorm.bias
        layer_model.fc1.weight = layer_rubert.intermediate.dense.weight
        layer_model.fc1.bias = layer_rubert.intermediate.dense.bias
        layer_model.fc2.weight = layer_rubert.output.dense.weight
        layer_model.fc2.bias = layer_rubert.output.dense.bias
        layer_model.final_layer_norm.weight = layer_rubert.output.LayerNorm.weight
        layer_model.final_layer_norm.bias = layer_rubert.output.LayerNorm.bias


def check_eq(inp_ids):
    rubert_embed = rubert.embeddings(inp_ids)
    model_embed = model.get_encoder().calc_embeddings(inp_ids)
    att_msk = inp_ids != tokenizer.pad_token_id
    assert torch.all(rubert_embed[att_msk] == model_embed[att_msk])

    rubert_enc, _ = rubert(inp_ids, attention_mask=att_msk)
    model_enc, _, _ = model.model.encoder(inp_ids, attention_mask=att_msk)
    if not torch.all(rubert_enc[att_msk] == model_enc[att_msk]):
        print('Not exactly equal (TODO why):')
        print(inp_ids)
        assert torch.max(torch.abs(rubert_enc[att_msk] - model_enc[att_msk])) < 1e-5

def check_equal():
    check_eq(torch.tensor([
        [101, 1000, 100, 999, 102],
        [101, 1001, 102, 0, 0],
    ]))
    check_eq(torch.tensor([
        [101, 1000, 100, 999, 102, 0],
        [101, 1001, 102, 0, 0, 0],
    ]))
    check_eq(torch.tensor([
        [101, 1000, 100, 999, 555],
        [101, 1001, 333, 0, 0],
    ]))
    check_eq(torch.tensor([
        [101, 1000, 100, 999, 555],
        [101, 1001, 102, 0, 0],
    ]))
    enc_txt = encode_text(tokenizer, test_text_sample, max_len=MAX_LEN_SRC)
    check_eq(enc_txt)
    check_eq(torch.cat([enc_txt] * 2))
    enc_txt_2 = torch.cat([enc_txt] * 2)
    enc_txt_2[1, 1:-1] = torch.randint(config.bos_token_id + 1, tokenizer.vocab_size, (enc_txt_2.shape[1] - 2,))
    enc_txt_2[1, -1] = tokenizer.convert_tokens_to_ids('[SEP]')
    check_eq(enc_txt_2)

check_equal()
torch.save(model.model.encoder.state_dict(), RUBART_ENC_WEIGHTS_DIR + 'encoder_state_dict.pth')
tokenizer.save_pretrained(RUBART_ENC_WEIGHTS_DIR)
config.save_pretrained(RUBART_ENC_WEIGHTS_DIR)

# check after reloading from disk
del model, tokenizer, config
tokenizer = BertTokenizer.from_pretrained(RUBART_ENC_WEIGHTS_DIR, do_lower_case=False)  # do_lower_case=False is crucial
config = BartConfig.from_pretrained(RUBART_ENC_WEIGHTS_DIR)
model = RuBartForConditionalGeneration(config).eval()
model.model.encoder.load_state_dict(torch.load(RUBART_ENC_WEIGHTS_DIR + 'encoder_state_dict.pth'))
check_equal()

# check whole model loading
del model, tokenizer, config
model, tokenizer = load_rubart_with_pretrained_encoder()
config = BartConfig.from_pretrained(RUBART_ENC_WEIGHTS_DIR)
model.eval()
check_equal()

