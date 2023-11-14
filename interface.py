import easyocr
import gradio as gr


reader = easyocr.Reader(['en'])
# This is a placeholder function for your image processing model.
# Replace this with your own model's prediction logic.
def process_image(file_path):
    result = reader.readtext(file_path)
    text = "Not Recognized"
    for item in result:
        if item[2] > 0.5 and ('1099-INT' in item[1] or '1099 - INT' in item[1]):
            text = '1099-INT'
            break

    if text == "Not Recognized":
        for item in result:
            if item[2] > 0.5 and ('W-2' in item[1] or 'W - 2' in item[1]):
                text = 'W-2'
                break

    return text

# Define the image input component
file_input = gr.File(type="filepath", label="Upload Image")

# Define the output text component
text_output = gr.Textbox(label="Result")

# Create the Gradio interface
interface = gr.Interface(fn=process_image,
                         inputs=file_input,
                         outputs=text_output,
                         title="Document classification",
                         description="Upload an image and get a description")

# Launch the app to allow users to interact with it
interface.launch()
