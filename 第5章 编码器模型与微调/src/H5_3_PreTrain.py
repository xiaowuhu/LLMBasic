import torch
import torch.nn as nn

from H5_3_Model_Together import BERTModel
from H5_3_PreTrainData import load_data
from H5_Helper import save_model, create_mask

def _get_batch_loss_bert(net, loss_func, vocab_size, tokens_X, segments_X,
                         mask_pos_X, mask_weight_X, mask_Y, nsp_Y, device):
    # 前向传播
    src_mask, src_key_padding_mask = create_mask(tokens_X, device, vocab['<pad>'])
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  src_mask, src_key_padding_mask,
                                  mask_pos_X)
    # 计算遮蔽语言模型损失
    mlm_loss = loss_func(mlm_Y_hat.reshape(-1, vocab_size), mask_Y.reshape(-1)) * mask_weight_X.reshape(-1)
    mlm_loss = mlm_loss.sum() / mask_weight_X.sum()
    # 计算下一句子预测任务的损失
    nsp_loss = loss_func(nsp_Y_hat, nsp_Y).mean()
    loss = mlm_loss + nsp_loss
    return mlm_loss, nsp_loss, loss

def validation(model, device, vocab, data_loader):
    print("--- validation ---")
    model.eval()
    total_loss, total_nsp_loss, total_mlm_loss = 0, 0, 0
    for tokens_X, segments_X, nsp_Y, mask_pos_X, mask_weight, mask_Y in data_loader:
        tokens_X = tokens_X.to(device)  #[N, L]
        segments_X = segments_X.to(device) #[N, L]
        mask_pos_X = mask_pos_X.to(device) #[N, 10] 10=64*0.15
        mask_weight_X = mask_weight.to(device) # 掩码 111110000
        mask_Y, nsp_Y = mask_Y.to(device), nsp_Y.to(device) # 预测token in mask
        src_mask, src_key_padding_mask = create_mask(tokens_X, device, vocab['<pad>'])
        hidden, mlm_Y_hat, nsp_Y_hat = model(tokens_X, segments_X,
                                  src_mask, src_key_padding_mask, mask_pos_X)
        mlm_loss = loss_func(mlm_Y_hat.reshape(-1, len(vocab)), mask_Y.reshape(-1)) * mask_weight_X.reshape(-1)
        mlm_loss = mlm_loss.sum() / mask_weight_X.sum()
        # 计算下一句子预测任务的损失
        nsp_loss = loss_func(nsp_Y_hat, nsp_Y).mean()
        total_mlm_loss += mlm_loss.item()
        total_nsp_loss += nsp_loss.item()
        total_loss += mlm_loss.item() + nsp_loss.item()
    print("total loss:%.4f"%(total_loss / len(data_loader)), "\tmlm loss:%.4f"%(total_mlm_loss / len(data_loader)), "\tnsp loss:%.4f"%(total_nsp_loss / len(data_loader)))
    return total_loss / len(data_loader)

def train_bert(train_iter, valid_iter, model, vocab_size, num_epoch, loss_func, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    running_loss = 0
    best_loss = 10
    log_step = 300
    for epoch in range(num_epoch):
        model.train()
        iteration = 0
        for tokens_X, segments_X, nsp_Y, mask_pos_X, mask_weight, mask_Y in train_iter:
            tokens_X = tokens_X.to(device)  #[N, L]
            segments_X = segments_X.to(device) #[N, L]
            mask_pos_X = mask_pos_X.to(device) #[N, 10] 10=64*0.15
            mask_weight = mask_weight.to(device) # 掩码 111110000
            mask_Y, nsp_Y = mask_Y.to(device), nsp_Y.to(device) # 预测token in mask
            optimizer.zero_grad()
            mlm_l, nsp_l, train_loss = _get_batch_loss_bert(
                model, loss_func, vocab_size, tokens_X, segments_X, 
                mask_pos_X, mask_weight, mask_Y, nsp_Y, device)
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()
            iteration += 1
            if iteration % log_step == 0:
                running_loss = running_loss / log_step
                print("iter: %d"%(iteration), "\trunning loss:%.4f"%(running_loss), "\tmlm loss:%.4f"%mlm_l.item(), "\tnsp loss:%.4f"%nsp_l.item())
                running_loss = 0
        val_loss = validation(model, device, vocab, valid_iter)
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, "Bert_Pretrain.pth")
        print("-----epoch =", epoch, "-----")

if __name__=="__main__":
    num_epoch, batch_size, max_len = 100, 128, 64
    train_iter, vocab = load_data(None, batch_size, max_len, name="train")
    valid_iter, _ = load_data(vocab, batch_size, max_len, name="valid")
    model = BERTModel(len(vocab), embed_size=128,
                    ffn_num_hiddens=256, 
                    num_heads=4, num_layers=2, 
                    max_len=max_len)
    loss_func = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_bert(train_iter, valid_iter, model, len(vocab), num_epoch, loss_func, optimizer)
