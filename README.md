# How to run
## Preparation 
1. Set up the keys for OpenAI and Google Gemini based on your needs.  <br />Set up *OPENAI_API_KEY* and *GOOGLE_API_KEY* in the env that the app will automatically load the keys.
2. Checkout the code to local repo
3. Open the mmconf.json to add or remove models based on your favorites. Make sure all the models configured here are supporting multimodal data or understing images at least. 
```json
{
  "ragModel":{
    "OpenAI":["gpt-4o" , "gpt-4-turbo"],
    "Google Gemini": [ "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"],
    "Ollama": ["llava-llama3" , "bakllava"]
  },
  "summaryModel":{
    "OpenAI": ["gpt-4o","gpt-4-turbo"],
    "Google Gemini": ["gemini-1.5-flash-latest" ,"gemini-1.5-pro-latest"],
    "Ollama": ["llava","llava-llama3","bakllava"]
  }
}
```
4. Create a python venv if necessary -- *Recommended*

## Run command
1. Install dependecies <br />
```bash
pip install -r requirements.txt
```
2. Start the app. Replace the python command based on the version on your machine <br />
```code
python3.12 -m  streamlit run demo.py
```

## Configuration before running
# Summarize the data 

<img width="1471" alt="截屏2024-05-31 16 40 43" src="https://github.com/melodylife/multimodalRAG/assets/2402592/f26a14f5-2cab-4686-a2c9-b031276a7b33">

Configure summary model options by selecting options from the boxes. 

# Start Q&A
<img width="1182" alt="截屏2024-05-31 16 51 31" src="https://github.com/melodylife/multimodalRAG/assets/2402592/2a3dc1e0-48f0-44be-a1f3-8f3e4856611c">
Once the data are succesfully processed configure the LLM model and relevant parameters for the RAG Q&A

