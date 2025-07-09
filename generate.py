import torch
from LLMZero import MiniGPT, Config
from LLMZero import Tokenizer


def prepare_context(prompt_tokens, context_len):
    padding_token_id = 27
    if len(prompt_tokens) < context_len:
        prompt_tokens = [padding_token_id] * config.context_len + prompt_tokens
    # kee the last context_len tokens
    prompt_tokens = prompt_tokens[-config.context_len :]

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
            input_tensor = prepare_context(input_tokens+generated_tokens, config.context_len)
            input_tensor = input_tensor.to(config.device)

            # generate a token
            logits, _ = model(input_tensor)
            next_token_logits = logits[:, -1, :]  # Get the logits for the last token
            next_token_id = torch.argmax(
                next_token_logits, dim=-1
            ).item()
            generated_tokens.append(next_token_id)

    print(f"Generated {len(generated_tokens)} tokens: {generated_tokens}")
    return tokenizer.decode(generated_tokens)


if __name__ == "__main__":
    # prepare the training configuration
    config, config_file = Config.from_latest_json("save/")
    print(f"Loaded config from {config_file}:")
    print(f"{config}")
    model = MiniGPT(config).to(config.device)
    # print(model)

    # Generate with the prompt
    prompt = "Hello, how are you?"
    generated_text = generate_text(model, config, prompt=prompt, max_length=10)
    print(f"Generated text: {generated_text}")
    print(f"Generated text length: {len(generated_text)}")
