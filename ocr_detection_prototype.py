import cv2 as cv
import pytesseract
from PIL import Image
import numpy as np

# Optional: Set path to tesseract executable (only if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 's' to capture screenshot and run OCR, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the live frame
    cv.imshow('Webcam Feed - Press s to capture', frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save screenshot
        screenshot_path = 'screenshot.png'
        cv.imwrite(screenshot_path, frame)
        print("Screenshot saved.")

        # Perform OCR using PyTesseract
        try:
            # Convert BGR (OpenCV) to RGB (Pillow/Tesseract)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Extract text
            text = pytesseract.image_to_string(pil_image)
            print("Extracted Text:")
            print("-" * 30)
            print(text)
            print("-" * 30)
        except Exception as e:
            print(f"OCR failed: {e}")

# Cleanup
cap.release()
cv.destroyAllWindows()
