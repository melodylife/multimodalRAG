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
