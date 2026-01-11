from transformers import AutoModelForCausalLM, AutoTokenizer

# load the model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
large_lang_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


def answer_question(query, embedded_data, top_k=3, max_tokens=512):
    # find the relevant information
    context = "\n Next Document: \n".join(embedded_data.retrieve_docs(query, k=top_k))

    # build the llm prompt
    prompt = f"""
You are a fact-based sports assistant. Answer the question using ONLY the information below. Do NOT hallucinate. Supply the answer as a human would do not address the instructions from the prompt. Do not repeat the prompt or context in your answer.
Context:
{context}
Question:
{query}
"""

    # query the model
    inputs = tokenizer(prompt, return_tensors="pt").to(large_lang_model.device)
    outputs = large_lang_model.generate(**inputs, max_new_tokens=max_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer
