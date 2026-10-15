The experimental setup and hyper-parameters of this repository are highlighted below:

encoder = ConformerEncoder(num_layers=4, d_model=512, d_ff=1024, num_heads=4, kernel_size=31, dropout=0.4, input_dim=512)
phoneme_decoder = PhonemeDecoder(vocab_size=500, d_model=512, hidden_size=512, num_layers=2, dropout=0.4) GRU based
cross_attention = MultiHeadAttention(d_model=512, num_heads=4, dropout=0.4, use_positional_encoding=False)
synthesizer = Synthesizer(d_model=512, rnn_dim=512, vocab_size=500   # Yoruba unit vocabulary size) LSTM based
optimizer = optim.Adam(model.parameters(), lr=1e-4) with a CosineAnnealingWithWarmrestart scheduler, incorporating an initial warmup of lr from 1% of lr upto the set lr for first 2 epochs.
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token, label_smoothing=0.1)
1680/420 datasplit, seq_len = 600

The repo consists of the saved metrics computed for the evaluated checkpoints.

