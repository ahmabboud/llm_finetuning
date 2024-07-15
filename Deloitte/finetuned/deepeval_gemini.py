# from langchain_google_vertexai import (
#     ChatVertexAI,
#     HarmBlockThreshold,
#     HarmCategory
# )
from langchain_google_genai import ChatGoogleGenerativeAI

from deepeval.models.base_model import DeepEvalBaseLLM
import os

# import google.generativeai as genai
from dotenv import load_dotenv

# load .env file
load_dotenv(override=True)
# os.environ["GEMINI_API_KEY"] =os.getenv('GEMINI_API_KEY')


# genai.configure(api_key=os.environ["GEMINI_API_KEY"])




class GoogleVertexAI(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Vertex AI Model"

# Initilialize safety filters for vertex model
# This is important to ensure no evaluation responses are blocked
# safety_settings = {
#     HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
# }



def main():

    # #TODO : Add values for project and location below
    # custom_model_gemini = ChatVertexAI(
    #     model_name="gemini-1.0-pro-002"
    #     , safety_settings=safety_settings
    #     , project= "<project-id>"
    #     , location= "<region>" #example : us-central1
    # )
    # Create the model
    # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 128,
    "response_mime_type": "text/plain",
    "max_new_tokens":128,
    }

    # model = genai.GenerativeModel(
    #   model_name="gemini-1.5-pro",#"gemini-1.5-flash",
    #   generation_config=generation_config,
    #   # safety_settings = Adjust safety settings
    #   # See https://ai.google.dev/gemini-api/docs/safety-settings
    # )

    model=ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=os.getenv('GEMINI_API_KEY'),max_output_tokens=500)

    # initiatialize the  wrapper class
    vertexai_gemini = GoogleVertexAI(model=model)
    print(vertexai_gemini.generate("Write me a joke"))

if __name__ == "__main__":
    main()
