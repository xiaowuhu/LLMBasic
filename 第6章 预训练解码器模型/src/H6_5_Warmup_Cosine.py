
import transformers
import torch
import os
import matplotlib.pyplot as plt

def test_scheduler(scheduler_select):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    config_filename = os.path.join(current_dir, "config/model_config_small.json")
    model_config = transformers.models.gpt2.GPT2Config.from_json_file(config_filename)
    model = transformers.models.gpt2.GPT2LMHeadModel(config=model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    if scheduler_select == 1:
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
    elif scheduler_select == 2:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100)
    else:
        raise

    lrs = []
    for i in range(1000):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])
    plt.plot(lrs)
    plt.grid()
    plt.show()

if __name__=="__main__":
    test_scheduler(1)
    test_scheduler(2)
