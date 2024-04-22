import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len  # 序列長度

        self.ds = ds  # 資料集
        self.tokenizer_src = tokenizer_src  # 來源語言的分詞器
        self.tokenizer_tgt = tokenizer_tgt  # 目標語言的分詞器
        self.src_lang = src_lang  # 來源語言代碼
        self.tgt_lang = tgt_lang  # 目標語言代碼

        # 特殊標記的設定
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)  # 起始標記
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)  # 結束標記
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)  # 填充標記

    def __len__(self):
        return len(self.ds)  # 資料集的大小

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]  # 來源文本
        tgt_text = src_target_pair['translation'][self.tgt_lang]  # 目標文本

        # 將文本轉換為索引序列
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # 計算需要添加的填充標記數量
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # 我們將添加 <s> 和 </s>
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # 我們僅在 decoder 上添加 <s>，在 label 添加 </s>

        # 確保填充標記數量不為負。如果是，則句子太長
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("句子太長")

        # 添加 <s> 和 </s> 標記
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # 僅添加 <s> 標記
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # 僅添加 </s> 標記
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # 雙重檢查張量的大小以確保它們都是 seq_len 長度
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
