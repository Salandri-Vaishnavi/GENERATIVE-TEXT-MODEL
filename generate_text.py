from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def clean_prompt(prompt):
    prompt = prompt.strip()
    prompt = re.sub(r'\s+', ' ', prompt)
    return prompt

def generate_text(prompt, max_length=150):
    instruction = (
        "Answer the question clearly and concisely:\n"
        "Q: What is machine learning?\n"
        "A: Machine learning is a method where computers learn from data without being explicitly programmed.\n\n"
        f"Q: {prompt}\nA:"
    )
    input_ids = tokenizer.encode(instruction, return_tensors='pt').to(device)

    output = model.generate(
        input_ids,
        max_length=max_length + input_ids.shape[-1],
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=2,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated[len(instruction):].strip()
    return answer

if __name__ == "__main__":
    user_prompt = input("Enter your topic or question: ")
    cleaned = clean_prompt(user_prompt)
    result = generate_text(cleaned)
    print("\n--- Generated Text ---\n")
    print(result)
