import torch
import torch.nn as nn
import math

"""
- 論文： Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf
- Youtuber： https://www.youtube.com/watch?v=ISNdQcPhsts&t=1599s
"""

class InputEmbeddings(nn.Module):
    """
    (頁碼 5) 3.4 嵌入和軟最大值
    將輸入的詞彙索引映射成詞嵌入向量，每個詞向量會乘以 sqrt(d_model) 來調整維度的規模。
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (批量, 序列長度) --> (批量, 序列長度, d_model)
        # 根據論文調整嵌入的規模
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    """
    (頁碼 6) 3.5 位置編碼
    為每個元素的嵌入添加關於其在序列中位置的信息，這使模型能夠利用序列元素的順序。在編碼階段，位置編碼確保即使輸入的單詞相同，它們在不同位置的表示也是唯一的，使得模型能夠根據單詞的不同位置進行不同的處理。
    使用正弦和餘弦函數生成位置編碼。
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # 創建形狀為 (序列長度, d_model) 的矩陣
        pe = torch.zeros(seq_len, d_model)
        # 創建形狀為 (序列長度) 的向量
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (序列長度, 1)
        # 創建形狀為 (d_model) 的向量
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # 在偶數索引處應用正弦
        pe[:, 0::2] = torch.sin(position * div_term) # sin(位置 * (10000 ** (2i / d_model))
        # 在奇數索引處應用餘弦
        pe[:, 1::2] = torch.cos(position * div_term) # cos(位置 * (10000 ** (2i / d_model))
        # 為位置編碼添加批量維度
        pe = pe.unsqueeze(0) # (1, 序列長度, d_model)
        # 將位置編碼註冊為緩衝區
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (批量, 序列長度, d_model)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    """
    (頁碼 3) 3.1 編碼器和解碼器堆棧
    1. 層正規化對每個特徵（列）分別計算均值和方差。
    2. 批正規化對每個樣本的所有特徵（行）計算均值和方差。
    """
    def __init__(self, features: int, eps:float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha 是一個可學習的參數
        self.bias = nn.Parameter(torch.zeros(features)) # 偏置是一個可學習的參數

    def forward(self, x):
        # x: (批量, 序列長度, 隱藏大小)
         # 保持維度以進行廣播
        mean = x.mean(dim=-1, keepdim=True) # (批量, 序列長度, 1)
        # 保持維度以進行廣播
        std = x.std(dim=-1, keepdim=True) # (批量, 序列長度, 1)
        # eps 是為了防止除以零或當 std 非常小的時候
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    """
    (頁碼 5) 3.3 位置智慧型前饋網絡
    非線性轉換：即使是 Transformer 模型中的注意力層能夠很好地處理序列數據，每個位置的輸出也只是輸入的線性組合。位置獨立的前饋網絡為模型引入了非線性，從而允許模型能夠學習更複雜的函數。
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 和 b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 和 b2

    def forward(self, x):
        # (批量, 序列長度, d_model) --> (批量, 序列長度, d_ff) --> (批量, 序列長度, d_model)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2
        return x


class MultiHeadAttentionBlock(nn.Module):
    """
    (P.4) 3.2.2 Multi-Head Attention
    注意力機制是一種用於確定輸入序列中哪些部分與輸出序列中最相關的技術。注意力分數是衡量輸入序列中每個元素與輸出序列中每個元素之間相關性的指標。
    將輸入分成多個頭，每個頭處理輸入的不同部分，這增加了模型的注意力彈性。
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of h
        # Make sure d_model is divisible by heads
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """(P.4) 3.2.1 Scaled Dot-Product Attention"""
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        x = self.w_o(x)
        
        return x

class ResidualConnection(nn.Module):
    """
    (P.3) 3.1 Encoder and Decoder Stacks - residual connection
    每個子層的輸入會加上其輸出，這有助於避免在深層網絡中的梯度消失問題。
    """
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    """
    包括一個自注意力塊和一個前饋網絡，以及它們各自的殘差連接。
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    """
    包含多個 EncoderBlock，每個 EncoderBlock 包括一個自注意力塊和一個前饋網絡，以及它們各自的殘差連接。
    """
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    使用多頭自注意力機制、交叉注意力機制和位置前饋網絡來進行處理。
    Self-Attention Block: 使用遮罩（mask）來防止位置向後查看（即在生成第 n 個元素時不能看到 n+1 及之後的元素）以了解已產生輸出的上下文關係。
    Cross-Attention Block: 也是一個多頭注意力塊，但它將解碼器的輸出與編碼器的輸出相結合。這使解碼器能夠專注於輸入序列中與當前生成的輸出最相關的部分。
    src_mask: 用來屏蔽輸入數據中可能包含一些填充（padding）元素
    tgt_mask: 確保解碼過程中的資訊流只來自於先前的輸出
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 解码器的自注意力层处理变长输入
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x

class Decoder(nn.Module):
    """
    包含多個 DecoderBlock
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    將解碼器的輸出從隱藏狀態空間投影到詞彙空間
    """
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        x = self.proj(x)
        x = torch.log_softmax(x, dim=-1)
        return x
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (批量, 序列長度, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (批量, 序列長度, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (批量, 序列長度, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    """
    函数允许我们以模块化的方式构建 Transformer 模型。通过这种方式，可以在一个函数中集中管理所有的构建步骤，包括初始化所有的层，设置他们的参数，
    """
    # 創建嵌入層
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # 創建位置編碼層
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # 創建編碼器塊
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # 創建解碼器塊
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # 創建編碼器和解碼器
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # 創建投影層
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # 創建 transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # 初始化參數
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


if __name__ == '__main__':
    import torch
    import torch.nn as nn

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if __name__ == '__main__':
        model = build_transformer(src_vocab_size=12448, tgt_vocab_size=12448, src_seq_len=4096, tgt_seq_len=4096, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048)
        total_params = count_parameters(model)
        print(f"Total trainable parameters: {total_params}") # 0.63B