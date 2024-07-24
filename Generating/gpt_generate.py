import pandas as pd
import json
from tqdm import tqdm
import openai
import time

openai.api_key = "YOUR_API_KEY"

ori_context = "Write an abstract in one paragraph and in a formal and academic style according to this title. " \
    + "Do not include any prefix and only keep the text of the abstract. "\
    + "Here is the title:\n\n{} " 
    
df_combined = pd.DataFrame()
corpus_abstracts = []
chunk_size = 12500
file_template = 'YOUR_INPUT_PATH'  
save_path = "YOUR_OUTPUT_PATH"
max_rows = 12500
processed_rows = 0  
results = []
instances_since_last_save = 0  
save_interval = 10

PBAR = tqdm(total=max_rows)

def save_results_to_file(results, save_path, mode='a'):
    """Save the current results to the file."""
    with open(save_path, mode, encoding="utf-8") as f:
        for ans in results:
            f.write(json.dumps(ans) + "\n")
    results.clear()  # Clear the results after saving


file_path = file_template

for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size):
    for index, row in chunk.iterrows():
        if index not in [1581, 3376, 3486, 3965, 4945, 8057, 9397, 9795, 11937, 12249]:
            PBAR.update(1)
            processed_rows += 1
            continue
        if processed_rows >= max_rows:
            break
        data = row.to_dict()
        try:
            corpusid = data["corpusid"]
            category = data["category"]
            title_info = data["content"]["annotations"]["title"]
            title_info_list = json.loads(title_info)
            if title_info_list:
                title_info = title_info_list[0]
                start_position = title_info["start"]
                end_position = title_info["end"]
                
                if not isinstance(start_position, int) or not isinstance(end_position, int):
                    print(f"Skipping row {index} due to non-integer positions")
                    continue
                
                title = data["content"]["text"][start_position:end_position]
                context = ori_context.format(title)
                
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",  # gpt-4
                        messages=[
                            {"role": "system", "content": "You are a helpful academic assistant."},
                            {"role": "user", "content": context}
                        ]
                    )
                    result = response['choices'][0]['message']['content']
                    results.append({"index": index, "corpusid": corpusid, "generate mode": "generated", "generated text": result, "category": category})
                except Exception as e:
                    print(f"Error: {e}")

                processed_rows += 1
                instances_since_last_save += 1
                PBAR.update(1)

                
                if instances_since_last_save >= save_interval:
                    save_results_to_file(results, save_path)
                    instances_since_last_save = 0  

        except KeyError as e:
            print(f"Missing data in row {index}: {e}")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    if processed_rows >= max_rows:
        break

PBAR.close()

# Save any remaining results
if results:
    save_results_to_file(results, save_path)

print(f"Completed. Saved the results for {processed_rows} instances.")
