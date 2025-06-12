import easyocr
import numpy as np
from PIL import Image

reader = easyocr.Reader(['en'])  # sekali inisialisasi saja

def ocr_plate_number(image: Image.Image) -> str:
    image_np = np.array(image)
    result = reader.readtext(image_np)
    text = ""

    if result:
        # Ambil hasil paling yakin
        text = result[0][-2]
    return text
