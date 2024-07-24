import pandas as pd
import json
from tqdm import tqdm
import openai
import time
from transformers import GPT2Tokenizer

openai.api_key = "YOUR_API_KEY"

ori_context = "Read the introduction and following full text of this research paper. " \
    + "Summarize this papar and write an abstract in one paragraph and in a formal and academic style. " \
    + "Do not include any prefix and only keep the text of the abstract. "\
    + "Here is the full text:\n\n{} " 


file_template = 'YOUR_INPUT_PATH'
save_path = "YOUR_OUTPUT_PATH"
failed_path = "YOUR_FAILED_PATH"
chunk_size = 12500

processed_rows = 0  
results = []
instances_since_last_save = 0  
save_interval = 10
max_rows = 12500

PBAR = tqdm(total = max_rows)


def save_results_to_file(results, save_path, mode='a'):
    """Save the current results to the file."""
    with open(save_path, mode, encoding="utf-8") as f:
        for ans in results:
            f.write(json.dumps(ans) + "\n")
    results.clear()

def extract_and_process_document(paragraphs_info, text):
    """Extract and process the document content based on paragraphs info."""
    if not paragraphs_info:
        return None
    start_of_first_paragraph = paragraphs_info[0]["start"]
    end_of_last_paragraph = paragraphs_info[-1]["end"]
    try:
        return text[start_of_first_paragraph:end_of_last_paragraph]
    except Exception as e:
        print(f"Error extracting document content: {e}")
        return None

file_path = file_template

for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size):
    for index, row in chunk.iterrows():
        # if index not in fail_list:
        #     continue
        if processed_rows >= max_rows:
            break
        data = row.to_dict()
     
        try:
            corpusid = data["corpusid"]
            category = data["category"]
            try:
                paragraphs_info = json.loads(data["content"]["annotations"]["paragraph"])
            except Exception as e:
                print(f"Error extracting paragraphs info: {e}")
                paragraphs_info = None
            document_text = extract_and_process_document(paragraphs_info, data["content"]["text"])
            
            if document_text == None:
                document_text = data["content"]["text"]
                print(f"Failed to extract paragraphs info for {index}, using the original text")
            
     
            context = ori_context.format(document_text)

            # average_token_length = 4
            # max_tokens = 2048
            # max_chars = max_tokens * average_token_length

            # if len(context) > max_chars:
            #     context = context[:max_chars]

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # gpt-4
                    messages=[
                        {"role": "system", "content": "You are a helpful academic assistant."},
                        {"role": "user", "content": context}
                    ]
                )
                result = response['choices'][0]['message']['content']
                results.append({"index": index, "corpusid": corpusid, "generate mode": "summarized", "generated text": result, "category": category})
                time.sleep(1)
              
            except:
                print(f"Failed to generate for {index}")    
                with open(failed_path, 'a', encoding="utf-8") as f:
                    f.write(json.dumps({"index": index, "corpusid": corpusid, "generate mode": "summarized", "generated text": "Failed", "category": category}) + "\n")                          
                time.sleep(1)

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

if results:
    save_results_to_file(results, save_path)

print(f"Completed. Saved the results for {processed_rows} instances.")
