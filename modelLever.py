#!/usr/local/bin/python3


from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import json
import os
import datetime
import base64
import uuid
import io

model_name = "modelLever"

def summaryImage(imgPath):
    with open(imgPath, "rb") as imgFile:
        encodedImg = base64.b64encode(imgFile.read())
        imgDecodeData = interpretImage(encodedImg.decode("utf-8"))
        return imgDecodeData


# generate multimodal prompt for Gemini and Ollama
def generatePrompt(importData):
    prompt = importData["promptData"]
    imageData = importData["imageData"]
    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{imageData}",
    }
    content_parts = []
    sys_parts = []
    text_part = {"type": "text", "text": prompt}
    content_parts.append(image_part)
    sys_parts.append(text_part)
    sys_prompt_msg = SystemMessage(content=sys_parts)
    return [sys_prompt_msg, HumanMessage(content=content_parts)]

# generate multimodal prompt template for OpenAI


def generateOpenAIImagePrompt():
    image_prompt_template = ImagePromptTemplate(
        input_variables=["imageData"],
        template={"url": "data:image/jpeg;base64,{imageData}"})
    sys_prompt_msg = SystemMessagePromptTemplate.from_template("{promptData}")
    promptTempwithImage = HumanMessagePromptTemplate(prompt=[image_prompt_template])
    return [sys_prompt_msg, promptTempwithImage]


# Create LLM model according to the options determined by users
def createModel():
    llm = {}
    modelSel = st.session_state.summaryModelSel
    if st.session_state.summaryService == "OpenAI":
        llm = ChatOpenAI(model=modelSel)
    elif st.session_state.summaryService == "Google Gemini":
        llm = ChatGoogleGenerativeAI(model=modelSel)
        # llm = VertexAI(model_name=modelSel)
    elif st.session_state.summaryService == "Ollama":
        llm = ChatOllama(model=modelSel)
    return llm


def interpretImage(imgEncBase64):
    prompt = "You are an experienced data analyst. You are collecting earning data of multiple companies. Read the metrics in the report carefully and extract the data relevant to financial performance of the company. Summarize the the content concisely."
    llm = createModel()

    if st.session_state.summaryService == "OpenAI":
        promptTempwithImage = generateOpenAIImagePrompt()
        chat_prompt_template = ChatPromptTemplate.from_messages(promptTempwithImage)
        # Use ChatOpenAI instead of OpenAI to leverage the vision model. Otherwise the call will fail definitely. Remember to resize the image before calling
        # OpenAI as OpenAI only accept the image size smaller than 2000x768 and larger than 512x512
        chain = chat_prompt_template | llm | StrOutputParser()
        response = chain.invoke({"imageData": imgEncBase64, "promptData": prompt})
        # print(response)
        return response
    chain = generatePrompt | llm | StrOutputParser()
    response = chain.invoke({"imageData": imgEncBase64, "promptData": prompt})
    return response



# Apply LLM to each of the set to summarize the contents.
def summarizeDatafromPDF(extractData):
    prompt = """You are an experienced data analyst. You are collecting earning data of multiple companies. Read the metrics in the format of text and tables. carefully and summarize the tables and text relevant to financial performance of the company concisely. Table or text content are : {dataContent}"""
    promptTemplate = ChatPromptTemplate.from_template(prompt)
    llm = createModel()
    summarizeChain = {"dataContent": lambda x: x} | promptTemplate | llm | StrOutputParser()
    # print(type(extractData["textElements"]))
    tableSummaries = []
    textSummaries = []
    for tbl in extractData["tableElements"]:
        response = summarizeChain.invoke(tbl)
        tableSummaries.append(response)
    for txt in extractData["textElements"]:
        response = summarizeChain.invoke(txt)
        textSummaries.append(response)

    imageSummaries = []
    for img in extractData["imgPath"]:
        imageBase64 = encodeImageBase64(img)
        chain = generatePrompt | llm | StrOutputParser()
        if st.session_state.summaryService == "OpenAI":
            promptTempwithImage = generateOpenAIImagePrompt()
            chat_prompt_template = ChatPromptTemplate.from_messages(promptTempwithImage)
            print("OpenAI...")
            chain = chat_prompt_template | llm | StrOutputParser()
        response = chain.invoke({"imageData": imageBase64, "promptData": "Please describe the image and summarize the content concisely"})
        # print(response)
        imageSummaries.append(response)
    print(f"The size of text summary is {len(textSummaries)}\n The size of table summary is {len(tableSummaries)}\n The size of image summary is  {len(imageSummaries)}\n")


