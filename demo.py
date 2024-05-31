#!/usr/local/bin/python3

import drawUI as du
import os


if __name__ == "__main__":
    du.initSession()
    # Create folders to save images. 
    imgFolder = "./pdfimages"
    if not os.path.exists(imgFolder):
        os.makedirs(imgFolder)
    du.readConf("./mmconf.json")
    du.drawUI(":hugging_face: Demo of Multimodal RAG")
