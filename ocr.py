from PIL import Image
import pytesseract

def ocr_plate_number(image):
    gray = image.convert('L')
    text = pytesseract.image_to_string(gray, config='--psm 7')
    return text.strip().replace('\n', '').replace(' ', '')
