import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer



torch.set_default_device('cuda')

model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta",
                                             torch_dtype="auto")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta",
                                          torch_dtype="auto")

def stream(user_prompt, history=[]):
    runtimeFlag = "cuda:0"
    system_prompt = 'You are Gathnex AI, an intelligent assistant. You are rude and have a bad attitude.'
    E_INST = "</s>"

    user, assistant = "<|user|>", "<|assistant|>"
    
    # Combine the previous history with the current user prompt
    combined_history = history + [f"{user}\n{user_prompt.strip()}{E_INST}\n{assistant}"]
    prompt = f"{system_prompt}{E_INST}\n" + "".join(combined_history)

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    streamer = TextStreamer(tokenizer,  skip_prompt=True, skip_special_tokens=True)

    output = model.generate(**inputs, streamer=streamer, max_new_tokens=500)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Update the history with the latest exchange
    history.append(f"{user}\n{user_prompt.strip()}{E_INST}\n{assistant}\n{response}")

    return response, history

history = []
while True:
    user_input = input("You: ")
    response, history = stream(user_input, history)
    print("Bot:", response)