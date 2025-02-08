import torch
from torch import nn
import os
import math
import collections

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))/size(y)


def save_model(model: nn.Module, name: str):
    print("---- save model... ----")
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    train_pth = os.path.join(current_dir, "model", name)
    torch.save(model.state_dict(), train_pth)

def load_model(model:nn.Module, name:str, device):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    model_pth = os.path.join(current_dir, "model", name)
    print("load model ", name)
    model.load_state_dict(torch.load(model_pth, map_location=device, weights_only=True))

def test_model(test_loader, model, device, loss_func):
    print("testing...")
    model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算，减少内存和计算资源浪费。
        for test_x, test_y in test_loader:
            x, y = test_x.to(device), test_y.to(device)
            predict = model(x)
            loss += loss_func(predict, y)
            pred = predict.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(y.data.view_as(pred)).cpu().sum().item()
 
    return loss.item()/len(test_loader), correct/len(test_loader.dataset)


def train_seq2seq(encoder: nn.Module, decoder: nn.Module, 
          train_data, test_data, 
          loss_func, checkpoint,
          encoder_optimizer, decoder_optimizer,
          num_epochs, device, save_model_names=None):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    best_loss = 10
    all_loss = []
    for epoch in range(num_epochs):
        running_loss = 0
        encoder.train()
        decoder.train()
        for step, (X, Y) in enumerate(train_data):
            X = X.to(device)
            Y = Y.to(device)
            _, encoder_hidden = encoder(X)
            decoder_output, _, _ = decoder(X.size(0), encoder_hidden, Y, device=device)
            # 32 x 10 x 2991 -> 320 x 2991
            output = decoder_output.view(-1, decoder_output.size(-1))
            # 32 x 10 -> 320
            target = Y.view(-1)
            # softmax, 从 2991 中取最大值作为输出
            step_loss = loss_func(output, target)
            running_loss += step_loss
            train_accu = accuracy(output, target)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            step_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            if (step + 1) % checkpoint == 0:
                loss = running_loss.item() / checkpoint
                print("epoch %d, iter %d, loss %.3f, accu %.3f" % (epoch, step, loss, train_accu))
                all_loss.append(loss)
                if save_model_names is not None and loss < best_loss:
                    best_loss = loss
                    save_model(encoder, save_model_names[0])
                    save_model(decoder, save_model_names[1])
                running_loss = 0



        if test_data is not None:
            test_loss, test_acc = test_model(test_data, encoder, device, loss_func)            
            if save_model_names is not None:
                if test_acc > best_acc:
                    best_acc = test_acc
                    save_model(encoder, save_model_names)
            print("---- epoch %d, test loss %.3f, test acc %.3f" % (epoch, test_loss, test_acc))
    return all_loss

def train_seq2seq_with_attention(encoder: nn.Module, decoder: nn.Module, 
          train_data, test_data, 
          loss_func, checkpoint,
          encoder_optimizer, decoder_optimizer,
          num_epochs, device, save_model_names=None):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    best_loss = 10
    all_loss = []
    for epoch in range(num_epochs):
        running_loss = 0
        encoder.train()
        decoder.train()
        for step, (X, Y) in enumerate(train_data):
            X = X.to(device)
            Y = Y.to(device)
            # X:32x10 -> o:32x10x128, h:1x32x128
            encoder_output, encoder_hidden = encoder(X)
            # o:32x10x128, h:1x32x128, Y:32x10 -> 32x10x2991
            decoder_output, _, _ = decoder(encoder_output, encoder_hidden, Y, device=device)
            # 32 x 10 x 2991 -> 320 x 2991
            output = decoder_output.view(-1, decoder_output.size(-1))
            # 32 x 10 -> 320
            target = Y.view(-1)
            # softmax, 从 2991 中取最大值作为输出
            step_loss = loss_func(output, target)
            running_loss += step_loss
            train_accu = accuracy(output, target)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            step_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            if (step + 1) % checkpoint == 0:
                loss = running_loss.item() / checkpoint
                print("epoch %d, iter %d, loss %.3f, accu %.3f" % (epoch, step, loss, train_accu))
                all_loss.append(loss)
                if save_model_names is not None and loss < best_loss:
                    best_loss = loss
                    save_model(encoder, save_model_names[0])
                    save_model(decoder, save_model_names[1])
                running_loss = 0

        if test_data is not None:
            test_loss, test_acc = test_model(test_data, encoder, device, loss_func)            
            if save_model_names is not None:
                if test_acc > best_acc:
                    best_acc = test_acc
                    save_model(encoder, save_model_names)
            print("---- epoch %d, test loss %.3f, test acc %.3f" % (epoch, test_loss, test_acc))
    return all_loss


def BLEU(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def generate_tril_mask(sz, DEVICE):
    mask = torch.ones((sz, sz), device=DEVICE) # 全1矩阵
    mask = torch.tril(mask) # 上三角矩阵
    mask.masked_fill_(mask == 0, float('-inf')) # 把 0 换成-inf
    mask.masked_fill_(mask == 1, float(0.0)) # 把True换成-inf, False换成0
    return mask

def create_mask(src, tgt, DEVICE, PAD):
    src_seq_len = src.shape[1]  # 64 x 6
    tgt_seq_len = tgt.shape[1]  # 64 x 5

    tgt_mask = generate_tril_mask(tgt_seq_len, DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_key_padding_mask = (src == PAD)
    tgt_key_padding_mask = (tgt == PAD)
    return src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask
