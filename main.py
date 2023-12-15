import numpy as np
import pytesseract
import cv2
import argparse
import utils


def find_rectangles(file_path, bi=False, lc=False, lr=False, hn=False):
    # Read the image
    image = cv2.imread(file_path)
    processed = image_preprocessing(image, bi, lc, lr, hn)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply edge detection
    edged = cv2.Canny(processed, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter for rectangles
    rectangles = []
    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the polygon is a rectangle
        if len(approx) == 4:
            # Check if the angles are approximately 90 degrees
            angles = []
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % 4][0]
                p3 = approx[(i + 2) % 4][0]
                angle = np.abs(
                    np.rad2deg(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])))
                if angle > 180:
                    angle = 360 - angle
                angles.append(angle)

            if all(40 <= angle <= 100 for angle in angles):
                rectangles.append(approx)

    # Extract and save rectangles
    max_area = 0
    largest_rect = None
    largest_crop = None
    for i, rect in enumerate(rectangles):
        x, y, w, h = cv2.boundingRect(rect)
        crop = image[y:y + h, x:x + w]
        area = w * h
        if area > max_area:
            max_area = area
            largest_rect = rect
            largest_crop = crop
    if largest_rect is not None:
        copy = image.copy()
        cv2.drawContours(copy, [largest_rect], -1, (0, 255, 0), 3)
        cv2.imshow('Image with Rectangle', copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return largest_crop


def find_plates(img):
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(img, config=config)
    return text


def process_plate(plate: np.ndarray) -> np.ndarray:
    gray = utils.get_grayscale(plate)
    processed = gray
    return processed


def image_preprocessing(image, bi, lc, lr, hn):
    processed = image.copy()
    if bi:
        processed = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    if lc:
        processed = cv2.equalizeHist(processed)
    if lr:
        scale = lr
        new_width = int(processed.shape[1] * scale)
        new_height = int(processed.shape[0] * scale)
        processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    if hn:
        processed = cv2.GaussianBlur(processed, (hn, hn), 0)
    return processed


def main():
    parser = argparse.ArgumentParser(description="Script para reconhecimento de placas veiculares.")
    parser.add_argument('-f', '--filepath', required=True, help="Caminho para o arquivo de imagem.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Ativa o modo verboso.")
    parser.add_argument('-bi', '--blured-image', action='store_true', help="Aplica o filtro de melhorar detalhes.")
    parser.add_argument('-lc', '--low-contrast', action='store_true',
                        help="Aplica o filtro de equalização de histograma.")
    parser.add_argument('-lr', '--low-resolution', action='store',
                        help="Aplica o filtro de redimensionamento com a escala passada")
    parser.add_argument('-hn', '--high-noise', action='store', help="Aplica o filtro de redução de ruído (3 ou 5).")
    args = parser.parse_args()

    filepath = args.filepath
    verbose = args.verbose
    lc = args.low_contrast
    bi = args.blured_image
    lr = args.low_resolution
    hn = args.high_noise
    if lr:
        lr = int(lr)
    if hn:
        hn = int(args.high_noise)

    plate = find_rectangles(filepath, bi, lc, lr, hn)
    print(find_plates(process_plate(plate)))


if __name__ == "__main__":
    main()
