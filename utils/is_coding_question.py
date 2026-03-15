import ollama

def is_coding_question(question, model="mistral"):
    prompt = f"""
Determine if the following question is about programming or can be solved by writing code.

Programming includes:
- debugging code
- writing code
- algorithms
- APIs
- frameworks
- coding errors
- software development
- system design

Question: {question}

Respond ONLY with YES or NO.
"""

    result = ollama.generate(model=model, prompt=prompt)
    decision = result["response"].lower()

    return "yes" in decision