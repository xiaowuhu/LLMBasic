
def test(vocab_size = None):
    from transformers import GPT2Model, GPT2LMHeadModel, AutoConfig, GPT2Tokenizer, AutoTokenizer

    if vocab_size is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        config = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer), n_ctx=1024)
        print(len(tokenizer))
    else:
        config = AutoConfig.from_pretrained("gpt2", vocab_size=vocab_size, n_ctx=1024)

    model = GPT2Model(config)
    model = GPT2LMHeadModel(config)
    print(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(model_size)
    print(f"GPT-2 size: {model_size/1000**2:.1f}M")


test(None)
test(6300)
test(9033)
