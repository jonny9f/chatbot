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
def stream(user_prompt):
    runtimeFlag = "cuda:0"
    system_prompt = 'You are Gathnex AI, an intelligent assistant. You are rude and have a bad attitude.'
    E_INST = "</s>"
    user, assistant = "<|user|>", "<|assistant|>"

    prompt = f"{system_prompt}{E_INST}\n{user}\n{user_prompt.strip()}{E_INST}\n{assistant}\n"

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    streamer = TextStreamer(tokenizer,  skip_prompt=True, skip_special_tokens=True)

    output = model.generate(**inputs, streamer=streamer, max_new_tokens=500)

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

while True:
    user_input = input("You: ")
    response = stream(user_input)
    print("Bot:", response)


