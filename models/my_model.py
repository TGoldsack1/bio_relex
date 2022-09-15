


class BartEncoderHybridGraphLayer(nn.Module):
    def __init__(self, config: BartConfig, ):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs



# ## Transformer Model

# torch.manual_seed(42)

# # INITIALIZATION
# encoder = nn.TransformerEncoder(
#   nn.TransformerEncoderLayer(
#     d_model=hdim, nhead=nhead, dim_feedforward=dim_feedforward, activation="gelu"
#   ),
#   num_layers=num_layers,
# ).to(device=device)
# encoder.eval()


# decoder = nn.TransformerDecoder(
#   nn.TransformerDecoderLayer(
#     d_model=hdim, nhead=nhead, dim_feedforward=dim_feedforward, activation="gelu"
#   ),
#   num_layers=num_layers,
# ).to(device=device)
# decoder.eval()



# def inference()
#   # INFERENCE LOOP
#   decoded_tokens = first_token
#   src_embeddings = encoder(src)
#   for i in range(lenoutput_to_decode):
#     mask_dec = generate_square_subsequent_mask(
#       i + 1, device=first_token.device
#     ) # create mask for autoregressive decoding
#     decoded_embeddings = embedding(decoded_tokens)

#     # the decoder uses the encoder output `src_embeddings`
#     output = decoder(decoded_embeddings, src_embeddings, tgt_mask=mask_dec)

#     logits = to_vocab(output) # projection to vocab size

#     # keep most likely tokens
#     top_indices = torch.argmax(logits, dim=-1)
#     # we only care about the last token that was decoded
#     top_indices_last_token = top_indices[-1:]
#     # add most likely token to the already decoded tokens
#     decoded_tokens = torch.cat(
#       [decoded_tokens, top_indices_last_token], dim=0
#     )



# class MyKnowledgeEnhancedModel(nn.Module):
#   def __init__(self, configs):
#     super(KnowledgeEnhancerModule, self).__init__()
#     self.configs = configs

#     self.cuid2embs = pickle.load(open(UMLS_EMBS, 'rb'))
#     print('Size of cuid2embs: {}'.format(len(self.cuid2embs)))

#     # Edge types of external knowledge graphs
#     self.ekg_etypes = set()
#     with open(UMLS_RELTYPES_FILE, 'r') as f:
#       for line in f:
#         self.ekg_etypes.add(line.strip().split('|')[1])

#     for rel in ['contains', 'has_title' 'has_keyword', 'was_published_in']:
#       self.ekg_etypes.add(rel)

#     self.ekg_etypes = list(self.ekg_etypes)
#     self.ekg_etypes.sort()