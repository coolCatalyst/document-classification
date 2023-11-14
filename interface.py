import easyocr
import gradio as gr


reader = easyocr.Reader(['en'])

flag = False
text = "Not Recognized"
# This is a placeholder function for your image processing model.
# Replace this with your own model's prediction logic.
def process_image(file):
    print("process_image")
    global flag, text
    if flag:
        return text
    flag = True
    # print("process_image called")
    # print(type(file))
    result = reader.readtext(file)
    # result = []
    print("file loaded")
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
    print(text)
    return text

def refresh():
    print("Refresh Button pressed")
    global flag, text
    flag = False
    text = "Not Recognized"

# Define the image input component
# file_input = gr.File(type="binary", label="Upload Image")

# # Define the output text component
# # text_output = gr.Textbox(label="Result")

# # process_btn = gr.Button("Process")
# # Create the Gradio interface
# interface = gr.Interface(fn=process_image,
#                          inputs=file_input, # gr.Button("Process file")],
#                          outputs="text")

# # Launch the app to allow users to interact with it
# interface.launch(server_name="0.0.0.0", server_port=7861, share=True)


with gr.Blocks() as demo:
    file_input = gr.File(type="binary", label="Upload Image")
    text_output = gr.Textbox(label="Result")
    process_btn = gr.Button("Process")
    refresh_btn = gr.Button("Refresh")
    process_btn.click(fn=process_image, inputs=file_input, outputs=text_output)
    refresh_btn.click(fn=refresh)

demo.launch(server_name="0.0.0.0", server_port=7861, share=True)