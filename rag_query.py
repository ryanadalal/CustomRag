from transformers import AutoModelForCausalLM, AutoTokenizer

# load the model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
large_lang_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


def answer_question(query, embedded_data, top_k=3, max_tokens=128):
    # find the relevant information
    info = embedded_data.retrieve_docs(query, k=top_k)
    context = "\n---\n".join(info["text"].tolist())

    prompt = f"""
You are a sports statistics assistant. Answer the question using only the information provided below. 
Do not invent or assume anything. Answer clearly, concisely, and in plain language. 
Do not reference the instructions or the context in your answer, so only output your final answer.

### Context (each game separated by ---):

{context}

### Question:

{query}

### Answer:
"""

    # query the model
    inputs = tokenizer(prompt, return_tensors="pt").to(large_lang_model.device)
    outputs = large_lang_model.generate(**inputs, max_new_tokens=max_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer
