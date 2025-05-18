def zipngram(text: str, ngram_size: int):
    """Helper function to generate n-grams from text."""
    words = text.lower().split() # Lowercase and split into words
    return zip(*[words[i:] for i in range(ngram_size)]) # Create n-grams

def repetition_penalty_reward(c):
    ngram_size = 3
    max_penalty = 1.0 # Maximum penalty for repetition

    ngrams = set() # Use a set to store unique n-grams
    total = 0
    for ng in zipngram(c, ngram_size): # Generate n-grams
        ngrams.add(ng) # Add n-gram to the set (duplicates are ignored)
        total += 1 # Count total n-grams

    # Calculate scaling factor: more repetition -> higher scaling
    scaling = 1 - len(ngrams) / total
    reward = scaling * max_penalty # Apply penalty based on scaling
    return reward

completions = [
    "This is a test.",
    "This is a test is a test is a test a test.",
]

for completion in completions:
    reward = repetition_penalty_reward(completion)
    print(reward)

