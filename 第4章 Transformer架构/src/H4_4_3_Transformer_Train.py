import torch
import torch.nn as nn
from H4_4_1_Transformer_Data import get_train_dataloader, PAD, get_test_dataloader
from H4_4_2_Transformer_Model import Seq2SeqTransformer
from H4_Helper import create_mask
from H4_4_4_Transformer_Test import evaluate

def train_epoch(model, optimizer, train_dataloader, test_dataloader):
    model.train()
    best_loss = 1
    for epoch in range(1, NUM_EPOCHS+1):
        print("----- epoch:", epoch)
        losses = 0
        running_loss = 0
        iter = 0
        for src, tgt in train_dataloader:
            src = src.to(DEVICE)  # N x Ls
            tgt = tgt.to(DEVICE)  # N x Lt

            tgt_input = tgt[:, :-1] # N x (Lt-1)
            # Ls x Ls Lt x Lt   N x Ls                N x Lt
            src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = create_mask(src, tgt_input, DEVICE, PAD)
            # N x Lt x V
            logits = model(src, tgt_input, 
                           src_mask, tgt_mask,
                           src_key_padding_mask = src_key_padding_mask, 
                           tgt_key_padding_mask = tgt_key_padding_mask, 
                           memory_key_padding_mask = src_key_padding_mask)

            optimizer.zero_grad()
            tgt_out = tgt[:, 1:].reshape(-1) # Lt-1
            # 把 [N, Lt-1, V] -> [-1, V]
            pred = logits.reshape(-1, logits.shape[-1]) # logits.shape[-1] == |Vt|
            loss = loss_fn(pred, tgt_out)
            loss.backward()

            optimizer.step()
            losses += loss.item()
            running_loss += loss.item()
            iter += 1
            if iter % 100 == 0:
                print("iter:", iter, "loss:", running_loss / 100)
                running_loss = 0

        loss = evaluate(model, test_dataloader, DEVICE)
        print("test loss =", loss)
        if loss < best_loss:
            best_loss = loss
            #save_model(model, "Transformer_10.pth")


if __name__=="__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64
    EMBED_SIZE = 512
    NHEAD = 8
    FC_HIDDEN = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    NUM_EPOCHS = 100
    torch.manual_seed(0)  # 保证词表顺序不变
    fr_vocab, en_vocab, train_dataloader = get_train_dataloader(BATCH_SIZE)
    test_dataloader = get_test_dataloader(fr_vocab, en_vocab, BATCH_SIZE)

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE,
                                 NHEAD, fr_vocab.n_words, en_vocab.n_words, FC_HIDDEN)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    train_epoch(transformer, optimizer, train_dataloader, test_dataloader)
