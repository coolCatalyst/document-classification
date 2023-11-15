import dotenv
import sys
import easyocr
from transformers import AutoModelForCausalLM, AutoTokenizer

dotenv.load_dotenv()
reader = easyocr.Reader(['en'])

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


prompt_template = """You need to classify the tax document into one of the following categories: 1099-INT, W-2
Following is words and numbers extracted from the tax document:
{combined_text}

###
Document type:

"""

def classify_llm(image_path):
    result = reader.readtext(image_path)

    # text = "Not Recognized"
    # for item in result:
    #     if item[2] > 0.5 and ('1099-INT' in item[1] or '1099 - INT' in item[1]):
    #         text = '1099-INT'
    #         break

    # if text == "Not Recognized":
    #     for item in result:
    #         if item[2] > 0.5 and ('W-2' in item[1] or 'W - 2' in item[1]):
    #             text = 'W-2'
    #             break
    
    combined_text = ', '.join([item[2] for item in result])
    prompt = prompt_template.format(combined_text=combined_text)
    model_inputs = tokenizer([prompt], return_tensors="pt")
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
    output = tokenizer.batch_decode(generated_ids)[0]
    # completion = openai.Completion.create(
    #     engine="text-davinci-003",
    #     max_tokens=1024,
    #     temperature=0,
    #     prompt=prompt
    # )
    
    # msg = completion.choices[0].text

    return output


if __name__ == "__main__":
    default_image_path = 'dataset/dataset/1099 - INT 2021.jpg'

    # Check if an argument was provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image_path
    
    result = classify_llm(image_path)
    print("Document type: ", result)