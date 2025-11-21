import json


with open('classification_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

temp=[]

for item in data:
    prompt = item["prompt"].strip()
    predicted_label = item["predicted_label"].strip()
    if predicted_label == "0" or predicted_label == "1":  
        temp.append({
            "prompt": prompt,
            "predicted_label": predicted_label
        })
        

with open('classification_results_filtered.json', 'w') as f:
    json.dump(temp, f, indent=4)


