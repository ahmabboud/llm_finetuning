# https://cheshirecat.ai/local-models-with-ollama/
# import gradio as gr
import os
import requests
from llama_cpp import Llama

llm_name = "Deloitte/fintuned/llama-3-8b-chat-doctor.gguf"
# llm_path = os.path.basename(llm_name)

# gguf_model = "Q4_K_M.gguf" # "Q6_K.gguf" 

# # download gguf model
# def download_llms(llm_name):
#     """Download GGUF model"""
#     download_url = ""
#     print("Downloading " + llm_name)
#     download_url = f"https://huggingface.co/MuntasirHossain/Meta-Llama-3-8B-OpenOrca-GGUF/resolve/main/{gguf_model}"

#     if not os.path.exists("model"):
#         os.makedirs("model")
    
#     llm_filename = os.path.basename(download_url)
#     llm_temp_file_path = os.path.join("model", llm_filename)

#     if os.path.exists(llm_temp_file_path):
#         print("Model already available")
#     else:
#         response = requests.get(download_url, stream=True)
#         if response.status_code == 200:
#             with open(llm_temp_file_path, 'wb') as f:
#                 for chunk in response.iter_content(chunk_size=1024):
#                     if chunk:
#                         f.write(chunk)
            
#             print("Download completed")
#         else:
#             print(f"Model download unsuccessful {response.status_code}")


# # define model pipeline with llama-cpp
# def initialize_llm(llm_model): 
#     model_path = ""
#     if llm_model == llm_name:
#         model_path = f"model/{gguf_model}"
#         download_llms(llm_model)
#     llm = Llama(
#         model_path=model_path,
#         n_ctx=1024, # input text context length, 0 = from model
#         n_threads=2,
#         verbose=False
#         )
#     return llm
    
llm = Llama(
        model_path=llm_name,
        n_ctx=1024, # input text context length, 0 = from model
        n_threads=2,
        verbose=False
        )

# format prompt as per the ChatML template. The model was fine-tuned with this chat template 
def format_prompt(input_text, history):
    system_prompt = """You are an expert and  helpful AI assistant. You are truthful and constructive in your response for real-world matters 
    but you are also creative for imaginative/fictional tasks."""
    prompt = ""
    if history:
        for previous_prompt, response in history:
            prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{previous_prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant"
    return prompt

# generate llm response
def generate(prompt, history, max_new_tokens=512): # temperature=0.95, top_p=0.9
    if not history:
        history = []

    # temperature = float(temperature)
    # top_p = float(top_p)

    kwargs = dict(
        # temperature=temperature,
        max_tokens=max_new_tokens,
        # top_p=top_p,
        stop=["<|im_end|>"]
    )

    formatted_prompt = format_prompt(prompt, history)

    # generate a streaming response 
    response = llm(formatted_prompt, **kwargs, stream=True)
    output = ""
    for chunk in response:
        output += chunk['choices'][0]['text']
        yield output
    return output

    # # generate response without streaming
    # response = llm(formatted_prompt, **kwargs)
    # return response['choices'][0]['text']

chatbot = gr.Chatbot(height=500)
with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as demo:
    gr.HTML("<center><h1>Fine-tuned Meta-Llama-3-8B Chatbot</h1><center>")
    gr.Markdown("This AI agent is using the MuntasirHossain/Meta-Llama-3-8B-OpenOrca-GGUF model for text-generation. <b>Note</b>: The app is running on a free basic CPU hosted on Hugging Facce Hub. Responses may be slow!")
    gr.ChatInterface(
        generate,
        chatbot=chatbot,  
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear",
        # description="This AI agent is using the MuntasirHossain/Meta-Llama-3-8B-OpenOrca-GGUF model for text-generation.",
        # additional_inputs=additional_inputs,
        examples=[["What is code vulnerability and how can Generative AI help to address code vulnerability?"], 
                  ["Imagine there is a planet named 'Orca' where life exists and the dominant species of the inhabitants are mysterious human-like intelligence. Write a short fictional story about the survival of this dominant species in the planet's extreme conditions. Use your imagination and creativity to set the plot of the story. Keep the story within 500 words."]]
    )
demo.queue().launch()