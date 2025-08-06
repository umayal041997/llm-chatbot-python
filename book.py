# library that allows us to download llm
from ctransformers import AutoModelForCausalLM

# llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", model_file="mistral-7b-instruct-v0.2.Q2_K.gguf")


def get_prompt(book_title: str, author:str) -> str:
    # System instruction
    system_instruction = (
        "You are a helpful AI assistant. For the given book and author, follow these steps:\n"
        "1. Provide a section titled 'Summary' with a short, concise 50‑word summary of the book.\n"
        "2. Provide a section titled 'Top 5 Quotes' listing the 5 most powerful quotes from the book.\n"
        "3. Provide a section titled 'Other Works by Author' listing other books or works by the same author.\n"
        "Make sure each section has a clear title and is separated by a blank line."
    )


    user_instruction = f"Book title: {book_title}\nAuthor: {author}"
    prompt = f"<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n{user_instruction} [/INST]"

    print("Final prompt:\n", prompt)
    return prompt


for word in llm(get_prompt("India that is Bharat", "Sai Deepak"), stream=True):
    print(word, end="", flush=True)