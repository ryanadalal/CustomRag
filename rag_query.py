from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# load the model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
large_lang_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
pipe = pipeline("text-generation", model=large_lang_model, tokenizer=tokenizer)


def answer_question(query, embedded_data, top_k=3, max_tokens=128):
    # find the relevant information
    info = embedded_data.retrieve_docs(query, k=top_k)
    context = "\n---\n".join(info["text"].tolist())

    prompt = f"""
Answer the question using ONLY the information in the context below.

Rules:
- Use team names exactly as written in the context.
- Do not add explanations or extra details.
- Do not infer anything not explicitly stated.
- Do not provide long explanations.
- If you do not find the answer in the context or are unsure, respond with "I don't know".
- Output ONE short sentence only.

Context:
{context}

Question:
{query}

Answer:
"""

    # query the model
    out = pipe(
        prompt,
        max_new_tokens=50,
        do_sample=False,
        return_full_text=False,
        top_p=1.0,
        top_k=50,
        temperature=1.0,
    )

    answer = out[0]["generated_text"].strip()
    return answer
