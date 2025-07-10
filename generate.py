import torch
from LLMZero import MiniGPT, Config
from LLMZero import Tokenizer


def prepare_context(prompt_tokens, context_len):
    # keep the last context_len tokens
    if len(prompt_tokens) > context_len:
        prompt_tokens = prompt_tokens[-context_len:]
    
    input_tensor = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0)
    return input_tensor


# generate text using the model
def generate_text(
    model: MiniGPT, config: Config, prompt: str = "Hi", max_length: int = 100
):
    # Tokenize the prompt
    tokenizer = Tokenizer(config.tokenizer_name)
    input_tokens = tokenizer.encode(prompt)
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = prepare_context(
                input_tokens + generated_tokens, config.context_len
            )
            input_tensor = input_tensor.to(config.device)

            # generate a token
            logits, _ = model(input_tensor)
            next_token_logits = logits[:, -1, :]  # Get the logits of the last token
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            if next_token_id == tokenizer.eot_token:
                break
            generated_tokens.append(next_token_id)

    print(f"Generated {len(generated_tokens)} tokens: {generated_tokens}")
    return tokenizer.decode(generated_tokens)


if __name__ == "__main__":
    # prepare the training configuration
    config, config_file = Config.from_latest_json("save/")
    print(f"Loaded config from {config_file}:")
    print(f"Loaded {config}")
    model = MiniGPT(config).to(config.device)
    model.eval()

    # Generate with the prompt
    prompt = "Once upon a time "
    generated_text = generate_text(model, config, prompt=prompt, max_length=100)
    print(f"Generated text length: {len(generated_text)}")
    print(f"Generated text: {generated_text}")
