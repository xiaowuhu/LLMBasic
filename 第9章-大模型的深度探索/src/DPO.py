
def dpo_loss(preferred_log_probs, rejected_log_probs, beta=0.1):
    """Computes the DPO loss function to optimize based on preferences"""
    return -torch.mean(torch.sigmoid(beta * (preferred_log_probs - 
    rejected_log_probs)))

def encode_text(prompt, response):
    """Encodes the prompt + response into tokenized format with proper padding"""
    tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue
    input_text = f"User: {prompt}\nAssistant: {response}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,         # Enable padding
        truncation=True,      # Truncate if too long
        max_length=512        # Set max length for safety
    )

    return inputs["input_ids"], inputs["attention_mask"]

loss_history = []  # Store loss values

optimizer = optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):  # Train for 10 epochs
    total_loss = 0

    for data in preference_data:
        prompt, preferred, rejected = data["prompt"], data["preferred"], 
        data["rejected"]

        # Encode preferred and rejected responses
        pref_input_ids, pref_attention_mask = encode_text(prompt, preferred)
        rej_input_ids, rej_attention_mask = encode_text(prompt, rejected)

        # Get log probabilities from the model
        preferred_logits = model(pref_input_ids, attention_mask=
        pref_attention_mask).logits[:, -1, :]
        rejected_logits = model(rej_input_ids, attention_mask=rej_attention_mask)
        .logits[:, -1, :]

        preferred_log_probs = preferred_logits.log_softmax(dim=-1)
        rejected_log_probs = rejected_logits.log_softmax(dim=-1)

        # Compute DPO loss
        loss = dpo_loss(preferred_log_probs, rejected_log_probs, beta=0.5)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    loss_history.append(total_loss)  # Store loss for visualization
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
