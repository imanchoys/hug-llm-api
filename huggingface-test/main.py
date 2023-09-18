import wikipediaapi
import requests

def query(payload, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/bert-large-uncased-whole-word-masking-finetuned-squad"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


wiki_wiki = wikipediaapi.Wikipedia('imanchoys', 'en')
x = 'Elon Musk'
page = wiki_wiki.page(x)
page_title = page.title
page_summary = page.summary[0:500]

model_id = "bert-large-uncased-whole-word-masking-finetuned-squad"
api_token = "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"
data = query({"question": f"who is {x}", "context": page_summary}, model_id, api_token)
print(data)