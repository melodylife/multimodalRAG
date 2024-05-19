#!/usr/local/bin/python3

import drawUI as du


if __name__ == "__main__":
    du.initSession()
    du.readConf("./mmconf.json")
    du.drawUI(":hugging_face: Demo of Multimodal RAG")
