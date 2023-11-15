import dotenv
import sys
import easyocr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
reader = easyocr.Reader(['en'])

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
# model.to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


prompt_template = """You need to classify the tax document into one of the following categories: 1099-INT, W-2
If user give you text that is extracted from document, you need to tell the document type.


Here are some examples

### Example 1 ###
User: 
Following is words and numbers extracted from the tax document:
BAXTER CREDIT UNION, 340 N MILWAUKEE AVE, VERNON HILLS, IL 60061, 800-388-7000, MDG2022 00009784 00, h"ilV Wu!"lll"hllual]Unh'llumGhllv, PATEL PRIYA, 1020 BETTE LANE, GLENVIEW, IL 60025, CORRECTED (if checked), Payer's RTN (optional), OMB No, 1545-0112, PAYER S name, street address, city, state, ZIP code, telephone no., Form, 1099-INT, Interest, BAXTER CREDIT UNION, 271992400, 340 N MILWAUKEE AVE, Interest Income, (Rev. January 2022), Income, VERNON HILLS, IL 60061, For calendar year, 800-388-7000, 109.56, 2021, 2 Early withdrawal penalty, Copy B, PAYER'S TIN, RECIPIENTS TN, 0.00, For Recipient, 3 Interest on U.S, Savings Bonds and Treasury obligations, 23-7250155, -4057, Q.00, RECIPIENTS name, Street address, city, state , and ZIP code, 4 Federal income tax withheld, 5 Investment expenses, This is important tax, S, 000, 0.00, information and is, being furnished to the, PATEL PRIYA, 6, Foreign tax paid, 7, Foreign country & U.S: possession, IRS. If you are, 1020 BETTE LANE, 0.00, required to file a, GLENVIEW; IL 60025, 8 Tax-exempt interest, 9 Specified private activity bond, return, a negligence, interest, penalty or other, sanction may be, 0.00, S, 0.00, imposed on you if, 10 Market dlscount, 11 Bond premium, this income is, taxable and the IRS, determines that it has, FATCA, not been, reported:, requirement, 12 Bond premium on Treasury obligations| 13 Bond premium on tax-exempt bond, 5, Account number (see instructions), 14 Tax-exempl and ax credit, 15 State, 16 State Identification no., 17 State tax withheld, bond CUSIP no., S, Acct#:  0004726335, Form 1099-INT (Rev. 1-2022), (keep for, records), www.irs gov/Form1O99INT, Department of the Treasury, Intemal Revenue Service, 0, 8, 1, 6, 5, 1, filing, your

Assistant:
1099-INT

### Example 2 ###
User:
Employee's social security number, Safe, accurate,, Visit the IRS website at, 336-96-1199, OMB No. 1545-0008, FASTI Use, (sev file, wwwirs_, govlefile, b Employer identification number (EIN), Wages, tips, other compensation, 2, Federal income tax withheld, 82-5251827, 57448.17, 3281.54, Employer's name; address; &nd ZIP code, 3, Social security wages, Social security tax withheld, Paramount Staffing Perm LLC, 57448.17, 3561.78, 1200 Shermer Rd, 5, Medicare wages and tips, Medicare tax withheld, Ste 300, 57448.17, 832.98, Northbrook; IL 60062, Social security tips, Allocated tips, 0.00, 0.00, Control number, 10, Dependent care benefits, 106331 119, 0.00, e, Employee's first name and initial, Last name, Suff_, 11, Nonqualified plans, 12a See instructions for box 12, 0.00, DD, 20353.99, Enrique Avila, 13, Statutory, Retirement, Third-party, 12b, 1708 W 35th St, empbyea, plan, sick pay, 8, 0.00, Apt 1, Chicago, IL 60609-1351, 14 Other, 12c, 8, 0.00, 12d, 3, 0.00, Employee's address and ZIP code, 15 State, Employer's state ID number, 16 State wages, tips; etc:, 17, State income tax, 18 Local wages; tips, etc, 19 Local income, 20 Locality name, W-2, Wage and Tax Statement, Department of the Treasury_Internal Revenue Service, Fom, Copy B_To Be Filed With Employee's FEDERAL Tax Return:, 2021, This information is being furnished to the Internal Revenue Service., tax

Assistant:
W-2

#####
User:
{text}

Assistant:

"""

def extract_last_response(conversation, assistant_prompt="Assistant:"):
    # Split the conversation into parts
    parts = conversation.split(assistant_prompt)
    # Check if conversation ends with the assistant's turn
    if len(parts) > 1:
        # Get last assistant's response
        last_response = parts[-1].split('</s>')[0].strip()
        return last_response
    else:
        return "Not sure"

def classify_llm(image_path):
    print("[INFO] OCR started")
    result = reader.readtext(image_path)
    print("[INFO] OCR END")
    
    combined_text = ', '.join([item[1] for item in result])
    prompt = prompt_template.format(text=combined_text)
    model_inputs = tokenizer([prompt], return_tensors="pt")  #.to(device)
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=4000, do_sample=True)
    conversation = tokenizer.batch_decode(generated_ids)[0]
    output = extract_last_response(conversation)
    return output


if __name__ == "__main__":
    default_image_path = 'dataset/dataset/suzanne and jim dase 2021 tax workpapers-10.jpg'

    # Check if an argument was provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image_path
    
    result = classify_llm(image_path)
    print("Document type: ", result)