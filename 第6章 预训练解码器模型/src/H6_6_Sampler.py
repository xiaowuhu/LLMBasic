import torch
import torch.nn as nn
from labml_nn.sampling import Sampler
from torch.distributions import Categorical

# Top-k Sampler
class TopKSampler(Sampler):
    # k is the number of tokens to pick
    # sampler is the sampler to use for the top-k tokens
    # sampler can be any sampler that takes a logits tensor as input and returns a token tensor; e.g. `TemperatureSampler`.
    def __init__(self, k: int, sampler: Sampler):
        self.k = k
        self.sampler = sampler

    # Sample from logits
    def __call__(self, logits: torch.Tensor):
        # New logits filled with −∞; i.e. zero probability
        zeros = logits.new_ones(logits.shape) * float('-inf')
        # Pick the largest k logits and their indices
        values, indices = torch.topk(logits, self.k, dim=-1)
        # Set the values of the top-k selected indices to actual logits.
        # Logits of other tokens remain −∞
        zeros.scatter_(-1, indices, values)
        # Sample from the top-k logits with the specified sampler.
        return self.sampler(zeros)


# top-p sampler
class NucleusSampler(Sampler):
    def __init__(self, p: float, sampler: Sampler):
        self.p = p
        self.sampler = sampler
        # Softmax to compute $P(x_i | x_{1:i-1})$ from the logits
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, logits: torch.Tensor):
        # Get probabilities $P(x_i | x_{1:i-1})$
        probs = self.softmax(logits)
        # Sort probabilities in descending order
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        # Get the cumulative sum of probabilities in the sorted order
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Find the cumulative sums less than $p$.
        nucleus = cum_sum_probs < self.p
        # Prepend ones so that we add one token after the minimum number
        # of tokens with cumulative probability less that $p$.
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
        # Get log probabilities and mask out the non-nucleus
        sorted_log_probs = torch.log(sorted_probs)
        sorted_log_probs[~nucleus] = float('-inf')
        # Sample from the sampler
        sampled_sorted_indexes = self.sampler(sorted_log_probs)
        # Get the actual indexes
        res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))
        #
        return res.squeeze(-1)


class TemperatureSampler(Sampler):
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor):
        # Create a categorical distribution with temperature adjusted logits
        dist = Categorical(logits=logits / self.temperature)
        # Sample
        #return dist.sample()
        return dist.logits, dist.probs, dist.sample()

if __name__=="__main__":
    import matplotlib.pyplot as plt

    logits = torch.tensor([3, 2, 1.5, 1.1, 1, 0.9, 0.5])
    temperatures = [0.2, 1, 2]
    torch.set_printoptions(precision=4, sci_mode=False)
    fig = plt.figure(figsize=(12,4))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ts_i = temperatures[i]
        print(f"t={ts_i} ---- ")
        ts = TemperatureSampler(ts_i)
        logits_t, probs_t, result_t = ts(logits)
        print(logits_t)
        print(probs_t)
        ax.bar(range(1,8), probs_t)
        ax.set_title("T=" + str(ts_i))
    plt.show()

    logits = torch.tensor([3, 2, 1.5, 1.1, 1, 0.9, 0.5])
    temperatures = [0.2, 1, 2]
    torch.set_printoptions(precision=4, sci_mode=False)
    fig = plt.figure(figsize=(12,4))
    for i in range(3):
        ts_i = temperatures[i]
        print(f"T={ts_i} ---- ")
        ts = TemperatureSampler(ts_i)
        results = []
        for j in range(10):
            _, _, result_t = ts(logits)
            results.append(result_t.item())
        print("token 序号：",results)

    temperature = 0.7
    logits = torch.tensor([3, 2, 1.5])
    ts = TemperatureSampler(temperature)
    logit, prob, result = ts(logits)
    print(logit, prob)
