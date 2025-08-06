# library that allows us to download llm
from ctransformers import AutoModelForCausalLM
import chainlit as cl


def get_prompt_orca(instruction: str, history: list[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the questions in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is the conversation history: {''. join(history)}. Now answer the question"
    prompt += f"{instruction}\n\n### Response:\n"
    # print(f"prompt create: {prompt}")   
    return prompt


def get_prompt_llama2(instruction: str, history: list[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the questions in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is the conversation history: {''. join(history)}. Now answer the question"
    prompt += f"{instruction}\n\n### Response:\n"
    # print(f"prompt create: {prompt}")   
    return prompt


def select_llm(model_name: str):
    global llm, get_prompt
    if model_name == "llama2":
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
        )
        get_prompt = get_prompt_llama2
        return "Model changed to Llama" 
    elif model_name == "orca":
        llm = AutoModelForCausalLM.from_pretrained(
            "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
        )
        get_prompt = get_prompt_orca
        return "Model changed to orca" 
    else:
        return "Model not found using old model"




@cl.on_message
async def on_message(message: cl.Message):
    if message.content.lower() in ["use llama2", "use orca"]:
        response = select_llm(message.content.lower().split()[1])
        await cl.Message(response).send()
        return
    if message.content.lower() == "forget everything":
        cl.user_session.set("message_history", [])
        await cl.Message("Uh oh, I've just forgotten our conversation history").send()
        return
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)
    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word
    await msg.update()
    message_history.append(response)
    # response = llm(prompt)
    # await cl.Message(response).send()


@cl.on_chat_start
async def on_chat_start():
    await cl.Message("Using model Llama....").send()
    select_llm("orca")
    cl.user_session.set("message_history", [])
    await cl.Message("Model loaded. How may I assis you!!").send()


# history = []
# question = "Which city is the capital of India?"
# answer = ""

# for word in llm(get_prompt(question), stream=True):
#     print(word, end="", flush=True)
#     answer += word
# print()

# history.append(answer)


# question = "And which is the capital city of the United states"
# for word in llm(get_prompt(question, history), stream=True):
#     print(word, end="", flush=True)
# print()

