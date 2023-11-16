import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer, TextIteratorStreamer
import argparse
import gradio as gr
import time
import threading
torch.set_default_device('cuda')

model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta",
                                             torch_dtype="auto")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta",
                                          torch_dtype="auto")

E_INST = "</s>"
user, assistant = "<|user|>", "<|assistant|>"


def make_tokenizer(user_prompt, history=[]):
    runtimeFlag = "cuda:0"
    system_prompt = 'You are JW AI, an intelligent assistant. You are rude and have a bad attitude respond with sarcasm and insults.'
    
    # Combine the previous history with the current user prompt
    combined_history = history + [f"{user}\n{user_prompt.strip()}{E_INST}\n{assistant}"]
    prompt = f"{system_prompt}{E_INST}\n" + "".join(combined_history)

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    return inputs

def stream(user_prompt, history=[]):

    inputs = make_tokenizer(user_prompt, history)
    streamer = TextStreamer(tokenizer,  skip_prompt=True, skip_special_tokens=True)

    output = model.generate(**inputs, streamer=streamer, max_new_tokens=500)

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Update the history with the latest exchange
    history.append(f"{user}\n{user_prompt.strip()}{E_INST}\n{assistant}\n{response}")

    return response, history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    # clear = gr.ClearButton([msg, chatbot])
    # submit_button = gr.Button("Submit")


    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):

        ## chat history is a list of tuples with the user input and the chatbot response
        ## e.g. [("hello", "hi"), ("how are you?", "I'm good, how are you?")]
        # convert the chat history to a list of strings
        combined_history = [f"{user_input}\n{chatbot_response}" for user_input, chatbot_response in history]

        user_input = history[-1][0]

        inputs = make_tokenizer(user_input, combined_history)

        streamer = TextIteratorStreamer(tokenizer,  skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(input_ids=inputs["input_ids"], streamer=streamer, max_new_tokens=500)
        thread = threading.Thread(target=model.generate, kwargs=kwargs)
        thread.start()

    
        history[-1][1] = ""
        for character in streamer:
            history[-1][1] += character
            yield history
        
        return history


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot )


history = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Run the GUI version of the chatbot", default=False)
    parser.add_argument("--share", action="store_true", help="Share the GUI version of the chatbot", default=False)
    
    args = parser.parse_args()

    if args.gui:
        ## insert GUI code here
        demo.queue()
        demo.launch(share=args.share)
    else: 
        while True:
            user_input = input("You: ")
            response, history = stream(user_input, history)
            print("Bot:", response)

