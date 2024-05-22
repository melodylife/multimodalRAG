#!/usr/local/bin/python3

import streamlit as st
import json
import dataLoader
import modelLever

module_name = "drawUI"


def readConf(confPath):
    with open(confPath, "r") as confFile:
        confData = json.load(confFile)
    st.session_state.confData = confData
    apiServices = [key for key in confData["models_in_service"].keys()]
    summaryModelServices = [key for key in confData["summaryModel"].keys()]
    st.session_state.apiServiceList = apiServices
    st.session_state.summaryModelService = summaryModelServices


def onChooseSummaryService():
    st.session_state.summaryModelDisabled = False
    st.session_state.uploaderDisabled = False
    st.session_state.summaryModelSelOptions = st.session_state.confData["summaryModel"][st.session_state.summaryService]


def serviceSelect():
    confData = st.session_state.confData
    if ("serviceSel" not in st.session_state) or bool(st.session_state["serviceSel"]):
        # if  bool(st.session_state["serviceSel"]):
        st.session_state.modelSelDisabled = False
        st.session_state.modelSelOptions = confData["models_in_service"][st.session_state["serviceSel"]]
    else:
        st.session_state.modelSelOptions = []
        st.session_state.modelSelDisabled = True


def initSession():
    # Control the usability of model selection
    if "modelSelDisabled" not in st.session_state:
        st.session_state.modelSelDisabled = True
    # Save the model selection to session state after selecting
    if "modelSelOptions" not in st.session_state:
        st.session_state.modelSelOptions = []
    # Initialize the llmAgent storage
    if "llmAgent" not in st.session_state:
        st.session_state.llmAgent = {}
    # Log the info to be presented
    if "serviceInfo" not in st.session_state:
        st.session_state.serviceInfo = ""
    # Log the message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Store the dataframe of uploaded report
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = {}
    if "apiServiceList" not in st.session_state:
        st.session_state.apiServiceList = []
    # Store the conf file
    if "confData" not in st.session_state:
        st.session_state.confData = {}
    # Store the vision model to parse data
    if "summaryModelService" not in st.session_state:
        st.session_state.summaryModelService = []
    # Usability of file uploader
    if "uploaderDisabled" not in st.session_state:
        st.session_state.uploaderDisabled = True
    # Usability of summary model selection
    if "summaryModelDisabled" not in st.session_state:
        st.session_state.summaryModelDisabled = True
    # Options of summary vision models
    if "summaryModelSelOptions" not in st.session_state:
        st.session_state.summaryModelSelOptions = []
    # Add a blob to store the retriever
    if "vectorretriever" not in st.session_state:
        st.session_state.vectorretriever = {}


def drawUI(title):
    st.title(title)
    #Initialize the chat window
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    #Create configuration side menu
    with st.sidebar:
    #Create blocks of options to process the pdf data
        st.selectbox(
            "Choose the approach to process PDF data",
            key="procAppr",
            options=["Convert PDF Page to Images",
                     "Extract data from PDF file"
                     ],
            index=1
        )
        st.divider()
        st.radio(
            "Choose the API service to process images",
            st.session_state.summaryModelService,
            key = "summaryService",
            on_change = onChooseSummaryService,
            index = None
        )
        st.selectbox(
            label="Please choose model to summarize images",
            options=st.session_state.summaryModelSelOptions,
            key="summaryModelSel",
            placeholder="Choose a model",
            disabled=st.session_state.summaryModelDisabled,
            index=None
        )
        pdfFile = st.file_uploader("Upload your pdf file  from here" , key = "fileuploader", type=["pdf"], disabled=st.session_state.uploaderDisabled)
        if (pdfFile is not None) and (st.session_state.summaryModelSel != "Please select the model") and (st.session_state.uploaderDisabled == False):
            with st.spinner("Processing PDF..."):
                dataLoader.processData(pdfFile)
                st.session_state.uploaderDisabled = True
        st.divider()
        # Below are the model service options of RAG query
        st.selectbox(
            label="Please select the model service",
            options=st.session_state.apiServiceList,
            key="serviceSel",
            index=None,
            on_change = serviceSelect,
            placeholder="Please choose the model service"
        )
        st.selectbox(
            label="Please select the model",
            options=st.session_state.modelSelOptions,
            disabled=st.session_state.modelSelDisabled,
            key="modelSel",
            placeholder="Please choose the model service"
        )
        tempOption = st.select_slider(
            label="Select the temperature",
            options=[0, 0.2, 0.5, 1],
            disabled=st.session_state.modelSelDisabled,
            key="tempSel"
        )
        st.divider()
        st.info(st.session_state.serviceInfo)
        st.button("Change the Model", type="primary")
    if prompt := st.chat_input("Please share what do you want to know..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                # trigger the agent chat call
                #response = "This is the test response. Please replace this line with meaningful agent call"
                response = modelLever.askLLM(prompt)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
