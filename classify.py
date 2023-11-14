import sys
import easyocr

reader = easyocr.Reader(['en'])

def classify_image(image_path):
    result = reader.readtext(image_path)

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


if __name__ == "__main__":
    default_image_path = 'dataset/dataset/1099 - INT 2021.jpg'

    # Check if an argument was provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image_path
    
    result = classify_image(image_path)
    print("Document type: ", result)