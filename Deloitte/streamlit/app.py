
#ref: https://huggingface.co/spaces/ysharma/Chat_with_Meta_llama3_8b/blob/main/app.py
import gradio as gr
import os
# import spaces
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
# Set an environment variable
# HF_TOKEN = os.environ.get("HF_TOKEN", None)


DESCRIPTION = '''
<div>
<h1 style="text-align: center;"> Mental Health Assistant Bot</h1>
<p>Meet MindMate, your compassionate and intelligent mental health assistant bot, designed to support your well-being and help you navigate the complexities of mental health. MindMate is here to offer personalized recommendations, activities, and resources to promote a healthier, happier you.</p>

</div>
'''

LICENSE = """
<p/>
---
Built with Meta Llama 3
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">Mental Health Assistant Bot</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Tell me what you are feeling...</p>
</div>
"""


css = """
h1 {
  text-align: center;
  display: block;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""
shared_space="/fs01/projects/fta_teams/deloitte"
merged_model= f"{shared_space}/merged_models/llama-3-8b-chat-doctor-merged-shyana-v1"
os.chdir("../../../../../../")
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(merged_model)
model = AutoModelForCausalLM.from_pretrained(merged_model, device_map="auto")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# @spaces.GPU(duration=120)
def chat_llama3_8b(message: str, 
              history: list, 
              temperature: float, 
              max_new_tokens: int
             ) -> str:
    """
    Generate a streaming response using the llama3-8b model.
    Args:
        message (str): The input message.
        history (list): The conversation history used by ChatInterface.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated response.
    """
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids= input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
    )
    # This will enforce greedy generation (do_sample=False) when the temperature is passed 0, avoiding the crash.             
    if temperature == 0:
        generate_kwargs['do_sample'] = False
        
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        #print(outputs)
        yield "".join(outputs)
        

# Gradio block
chatbot=gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Gradio ChatInterface')

with gr.Blocks(fill_height=True, css=css) as demo:
    
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    gr.ChatInterface(
        fn=chat_llama3_8b,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Slider(minimum=0,
                      maximum=1, 
                      step=0.1,
                      value=0.95, 
                      label="Temperature", 
                      render=False),
            gr.Slider(minimum=128, 
                      maximum=4096,
                      step=1,
                      value=512, 
                      label="Max new tokens", 
                      render=False ),
            ],
        examples=[
            ["I've been feeling very depressed lately. What can I do to start feeling better?"],
            ["I have trouble falling asleep and staying asleep. How can I improve my sleep habits?"],
            ["I'm experiencing a lot of anxiety during the day. What techniques can help me manage this?"],
            ["I feel overwhelmed by stress at work. What strategies can I use to cope?"],
            ["I've lost interest in activities I used to enjoy. Is this a normal part of my condition?"],
            ["I find it difficult to get out of bed and start my day. How can I motivate myself?"],
            ["I feel like I'm constantly worried about everything. How can I reduce my general anxiety?"],
            ["I feel like I don't have any control over my life. How can I regain a sense of control?"]
            ],
        cache_examples=False,
                     )
    
    gr.Markdown(LICENSE)
    
if __name__ == "__main__":
    demo.launch(share=True)
    
