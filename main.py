import numpy as np
import pytesseract

import cv2
from pytesseract import Output

import argparse

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def main():
    parser = argparse.ArgumentParser(description="Script para reconhecimento de placas veiculares.")
    parser.add_argument('-f', '--filepath', required=True, help="Caminho para o arquivo de imagem.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Ativa o modo verboso.")
    args = parser.parse_args()

    filepath = args.filepath
    verbose = args.verbose

    if verbose:
        print(f"Modo verboso ativado. Processando arquivo: {filepath}")

    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


    if image is None:
        print("Erro ao carregar a imagem")
        exit()
    if verbose:
        cv2.imshow("Gray x Equalizado x Thresh", np.hstack([gray, equalized, thresh]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    custom_config = r'--psm 6'
    print(pytesseract.image_to_string(image, config=custom_config))
    print(pytesseract.image_to_string(thresh, config=custom_config))
    print(pytesseract.image_to_string(gray, config=custom_config))
    print(pytesseract.image_to_string(equalized, config=custom_config))



if __name__ == "__main__":
    main()
