#!/usr/local/bin/python3

import streamlit as st
import json
import os
import pandas as pd
import matplotlib
import datetime
import base64
import uuid
import fitz
import io
from pdf2image import convert_from_bytes
import modelLever

module_name = "dataLoader"

# Extract data from PDFs and categorize them into sets of text, table and image respectively.

def ExtractDataFromPDF(pdfFileContent):
    # Extract elements from PDF file
    pdfLoader = fitz.open("pdf", io.BytesIO(pdfFileContent))
    textElements = []
    tableElements = []
    imgPath = []
    for pageIndex in range(len(pdfLoader)):
        pageContent = pdfLoader[pageIndex]

        # Extract Text
        text = pageContent.get_text()
        textElements.append(text)
        # Extract Tables
        tables = pageContent.find_tables()
        for table in tables:
            # Save the content of table in the csv format in the list
            tableElements.append(table.extract())
        # Extract Images
        for imgIndex, imgData in enumerate(pageContent.get_images(), start=1):
            xref = imgData[0]
            imgData = pdfLoader.extract_image(xref)
            imgByte = imgData["image"]
            imgExt = imgData["ext"]
            imgContent = Image.open(io.BytesIO(imgByte))
            imgContent.save(open(f"./pdfimages/pdfImage{pageIndex + 1}_{imgIndex}.{imgExt}", "wb"))
            imgPath.append(f"./pdfimages/pdfImage{pageIndex + 1}_{imgIndex}.{imgExt}")
    print(f"{len(tableElements)} Tables\n{len(textElements)} Text Passages\n{len(imgPath)} Images")
    return {"textElements": textElements, "tableElements": tableElements, "imgPath": imgPath}


# Convert PDF pages into images

def ConvertPDFtoImages(pdfFileContent):
    images = convert_from_bytes(pdfFileContent, size=(768, 768))
    imageSummary = []
    for i in range(len(images)):
        fileName = "./pdfimages/page" + str(i) + ".jpg"
        images[i].save(fileName, 'JPEG')
        modelLever.summaryImage(fileName)
        imageSummary.append(imageSummary)
    st.session_state.imageSummary = imageSummary
    sizeofsummary = len(imageSummary)
    print(sizeofsummary)


# Start loading data from PDF files on the upon uploading.
def processData(pdfFile):
    if st.session_state.procAppr == "Extract data from PDF file":
        print("Extract Data")
        extractData = ExtractDataFromPDF(pdfFile.getvalue())
        modellever.summarizeDatafromPDF(extractData)
    elif st.session_state.procAppr == "Convert PDF Page to Images":
        print("Convert PDF")
        ConvertPDFtoImages(pdfFile.getvalue())