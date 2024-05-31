#!/usr/local/bin/python3


from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.globals import set_verbose
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
import re

model_name = "modelLever"


def summaryImage(imgPath):
    with open(imgPath, "rb") as imgFile:
        encodedImg = base64.b64encode(imgFile.read())
        imgDecodeData = interpretImage(encodedImg.decode("utf-8"))
        return imgDecodeData

def generatePromptwithImageList(importData):
    ctxDataList = importData["txtData"]
    imageList = importData["imageData"]
    prompt = importData["promptData"]
    query = f"Answer the question only based on the information extracted from the text and images.Answer the question concisely. Question: {prompt}"
    print(f"The length of image list {len(imageList)}. The length of text list is {len(ctxDataList)}\n")
    content_parts = []
    content_parts.append({"type": "text", "text": query})
    for imgData in imageList:
        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{imgData}",
        }
        content_parts.append(image_part)
    for txtData in ctxDataList:
        text_part = {"type": "text", "text": txtData}
        content_parts.append(text_part)
    return [HumanMessage(content=content_parts)]

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
    return [HumanMessage(content=content_parts)]

# generate multimodal prompt template for OpenAI
def generateOpenAIImagePrompt():
    image_prompt_template = ImagePromptTemplate(
        input_variables=["imageData"],
        template={"url": "data:image/jpeg;base64,{imageData}"})
    sys_prompt_msg = SystemMessagePromptTemplate.from_template("{promptData}")
    promptTempwithImage = HumanMessagePromptTemplate(prompt=[image_prompt_template])
    return [sys_prompt_msg, promptTempwithImage]

def generateOpenAIPromptwithImageList(importData):
    prompt = importData["promptData"]
    ctxDataList = importData["txtData"]
    imageData = importData["imageData"]
    msgContent = []
    query = f"Answer the question only based on the information extracted from the text ,mages and tables.Answer the question concisely. Question: {prompt}"
    msgContent.append({"type": "text", "text": query})
    for img in imageData:
        msgContent.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        )
    for txtData in ctxDataList:
        msgContent.append({"type": "text", "text": txtData})
    return [HumanMessage(content=msgContent)]

# Create LLM model according to the options determined by users


def createModel(modelService , model , temp):
    # The temperature by default Google 0.7, OpenAI 0.7, Ollama 0.8. For summary pipeline, I set them all to 0.8. But for the Q&A pipeline, you can set up by dragging the bar
    llm = {}
    #modelSel = st.session_state.summaryModelSel
    modelSel = model
    if modelService == "OpenAI":
        llm = ChatOpenAI(model=modelSel , temperature = temp)
    elif modelService == "Google Gemini":
        llm = ChatGoogleGenerativeAI(model=modelSel , temperature = temp)
        # llm = VertexAI(model_name=modelSel)
    elif modelService == "Ollama":
        llm = ChatOllama(model=modelSel , temperature = temp)
    return llm


def interpretImage(imgEncBase64):
    prompt = "You will be presented with multiple file pages. The pages will be including texts, tables ,chart and images. Describe the page contents by extracting the information from it and summarize the the content with key information like name, subject, metrics, chart, table content from it."
    # For the summarization, temperature is set to 0.8 by default
    llm = createModel(st.session_state.summaryService , st.session_state.summaryModelSel , 0.8)
    if st.session_state.summaryService == "OpenAI":
        promptTempwithImage = generateOpenAIImagePrompt()
        chat_prompt_template = ChatPromptTemplate.from_messages(promptTempwithImage)
        # Use ChatOpenAI instead of OpenAI to leverage the vision model. Otherwise the call will fail definitely. Remember to resize the image before calling
        # OpenAI as OpenAI only accept the image size smaller than 2000x768 and larger than 512x512
        chain = chat_prompt_template | llm | StrOutputParser()
        response = chain.invoke({"imageData": imgEncBase64, "promptData": prompt})
        return response
    chain = generatePrompt | llm | StrOutputParser()
    response = chain.invoke({"imageData": imgEncBase64, "promptData": prompt})
    return response


# Apply LLM to each of the set to summarize the contents.
def summarizeDatafromPDF(extractData):
    prompt = """You are an assistant tasked with summarizing tables, text and images. Summarize the content from table, text and image chunks. Pay attention to the term definition, time period, numbers, list, all the key points, etc.  Table or text content are : {dataContent}"""
    promptTemplate = ChatPromptTemplate.from_template(prompt)
    # For the summarization, temperature is set to 0.8 by default
    llm = createModel(st.session_state.summaryService , st.session_state.summaryModelSel , 0.8)
    # Create chain to summarize the text data
    summarizeChain = {"dataContent": lambda x: x} | promptTemplate | llm | StrOutputParser()
    # print(type(extractData["textElements"]))
    tableSummaries = []
    textSummaries = []
    for tbl in extractData["tableElements"]:
        print(f"here's the table {tbl}\n")
        response = summarizeChain.invoke(tbl)
        print(f"here's the table summary {response}\n")
        tableSummaries.append(response)
    for txt in extractData["textElements"]:
        # print(f"here's the text {txt}\n")
        response = summarizeChain.invoke(txt)
        textSummaries.append(response)
    imageSummaries = []
    for img in extractData["imgPath"]:
        imageBase64 = encodeImageBase64(img)
        chain = generatePrompt | llm | StrOutputParser()
        if st.session_state.summaryService == "OpenAI":
            promptTempwithImage = generateOpenAIImagePrompt()
            chat_prompt_template = ChatPromptTemplate.from_messages(promptTempwithImage)
            chain = chat_prompt_template | llm | StrOutputParser()
        response = chain.invoke({"imageData": imageBase64, "promptData": "Please describe the image and summarize the content concisely"})
        # print(response)
        imageSummaries.append(response)
    print(f"The size of text summary is {len(textSummaries)}\n The size of table summary is {len(tableSummaries)}\n The size of image summary is  {len(imageSummaries)}\n")
    return {"textSummaries": {"mediatype": "text", "payload": extractData["textElements"], "summary": textSummaries},
            "tableSummaries": {"mediatype": "text", "payload": extractData["tableElements"], "summary": tableSummaries},
            "imageSummaries": {"mediatype": "image", "payload": extractData["imgPath"], "summary": imageSummaries}}


# Create the vectore storage and retriever for the RAG data retriever
def retrieverGenerator(summarizedData):
    # Create the retriever and vectore DB for use
    vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
    store = InMemoryStore()
    id_key = "rec_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key
    )
    # Extract the structured data from former function
    for key in summarizedData.keys():
        mediaType = summarizedData[key]["mediatype"]
        summary = summarizedData[key]["summary"]
        payload = summarizedData[key]["payload"]
        print(f"size of mediatype {len(mediaType)}. The size of summary {len(summary)}. The size of payload {len(payload)}")
        docs_ids = [str(uuid.uuid4()) for _ in summary]

        # To avoid any empty list to crash the program
        if (len(summary) == 0):
            continue
        if (mediaType == "text"):
           summaryDoc = [
               Document(page_content=s, metadata={id_key: docs_ids[i], "mediaType": mediaType})
                for i, s in enumerate(summary)
            ]
        elif (mediaType == "image"):
           summaryDoc = [
               Document(page_content=s, metadata={id_key: docs_ids[i], "mediaType": mediaType, "source": payload[i]})
                for i, s in enumerate(summary)
            ]
        # Add summary to the vectore store and add the original data to the in memory storage
        retriever.vectorstore.add_documents(summaryDoc)
        retriever.docstore.mset(list(zip(docs_ids, payload)))
    # Push retriever into the web context for the call by other functions
    st.session_state.vectorretriever = retriever


def askLLM(query):
    retriever = st.session_state.vectorretriever
    searchDocs = retriever.vectorstore.similarity_search(query)
    #print(f"This is the vector search result {searchDocs[0]}\n ")
    imageData = []
    txtData = []
    relevantImages = "<br /><br />  <h2>Below are the relevant images retrieved</h2>"
    for doc in searchDocs:
        rec_id = doc.metadata["rec_id"]
        mediaType = doc.metadata["mediaType"]
        ctxContent = retriever.docstore.mget([rec_id])
        print(f"This is the record content {rec_id}\n   {ctxContent}\n")
        if(mediaType == "text"):
            txtData.append(ctxContent[0])
        elif(mediaType == "image"):
            imgB64Enc = encodeImageBase64(ctxContent[0])
            imageData.append(imgB64Enc)
            relevantImages = relevantImages + f"<br /><br /><img   width=\"60%\" height=\"30%\"  src=\"data:image/jpeg;base64,{imgB64Enc}\">  "
    llmModel = createModel(st.session_state.serviceSel , st.session_state.modelSel , st.session_state.tempSel)
    chain = {}
    modelService = st.session_state.serviceSel
    modelSelected = st.session_state.modelSel
    queryPrompt = "Answer the question only according to the content in the provided context including  images, texts and tables. Output the answer in the format of markdown. Please say I don't know if there's no relevant content in the image. \n\nQuestion: " + query
    if (modelService == "OpenAI"):
        chain = generateOpenAIPromptwithImageList | llmModel | StrOutputParser()
    else:
        chain = generatePromptwithImageList | llmModel | StrOutputParser()
    response = chain.invoke({"imageData": imageData, "txtData": txtData , "promptData": query})
    if (len(imageData) == 0):
        relevantImages = ""
    return response + relevantImages

def encodeImageBase64(imgPath):
    with open(imgPath, "rb") as imgContent:
        base64Data = base64.b64encode(imgContent.read())
        return base64Data.decode("utf-8")
