import cv2 as cv
import pytesseract
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Optional: Set path to tesseract executable (only if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_for_ocr(frame):
    """
    Preprocess image for better OCR accuracy
    Returns a cleaned, binary image optimized for text recognition
    """
    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Denoise the image
    denoised = cv.fastNlMeansDenoising(contrast, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Apply Otsu's binarization for thresholding
    _, binary = cv.threshold(denoised, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Alternative: Adaptive thresholding (uncomment if Otsu doesn't work well)
    # binary = cv.adaptiveThreshold(denoised, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv.THRESH_BINARY, 11, 2)

    # Optional: Morphological operations to clean up noise
    kernel = np.ones((1, 1), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    # Resize to make text larger (helps OCR significantly)
    scale_percent = 200  # 200% of original size
    width = int(binary.shape[1] * scale_percent / 100)
    height = int(binary.shape[0] * scale_percent / 100)
    resized = cv.resize(binary, (width, height), interpolation=cv.INTER_CUBIC)

    return resized

def validate_phred33(text):
    """
    Validate and clean extracted text to only include valid PHRED+33 characters
    PHRED+33 uses ASCII characters from 33 (!) to 126 (~)
    """
    valid_chars = set(chr(i) for i in range(33, 127))
    # Remove newlines and spaces, keep only valid PHRED characters
    cleaned = ''.join(c for c in text if c in valid_chars)
    return cleaned

def phred33_to_quality_scores(quality_string):
    """
    Convert PHRED+33 encoded string to numeric quality scores
    Quality Score = ASCII value - 33
    """
    return [ord(char) - 33 for char in quality_string]

def plot_quality_scores(quality_scores, quality_string):
    """
    Create a graph of quality scores
    """
    if not quality_scores:
        print("No quality scores to plot!")
        return

    positions = list(range(1, len(quality_scores) + 1))

    plt.figure(figsize=(12, 6))

    # Plot quality scores
    plt.subplot(2, 1, 1)
    plt.plot(positions, quality_scores, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.axhline(y=20, color='r', linestyle='--', label='Q20 threshold (99% accuracy)')
    plt.axhline(y=30, color='g', linestyle='--', label='Q30 threshold (99.9% accuracy)')
    plt.xlabel('Position in Read')
    plt.ylabel('Quality Score')
    plt.title('PHRED Quality Scores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, max(quality_scores) + 5 if quality_scores else 50)

    # Show the quality string
    plt.subplot(2, 1, 2)
    plt.text(0.5, 0.5, f'Quality String:\n{quality_string}',
             ha='center', va='center', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('quality_scores_graph.png', dpi=150, bbox_inches='tight')
    print("Graph saved as 'quality_scores_graph.png'")
    plt.show()

def print_quality_statistics(quality_scores):
    """
    Print statistical information about the quality scores
    """
    if not quality_scores:
        print("No quality scores available!")
        return

    print("\n" + "=" * 50)
    print("QUALITY SCORE STATISTICS")
    print("=" * 50)
    print(f"Total positions: {len(quality_scores)}")
    print(f"Mean quality score: {np.mean(quality_scores):.2f}")
    print(f"Median quality score: {np.median(quality_scores):.2f}")
    print(f"Min quality score: {min(quality_scores)}")
    print(f"Max quality score: {max(quality_scores)}")
    print(f"Std deviation: {np.std(quality_scores):.2f}")

    # Count bases above quality thresholds
    q20_count = sum(1 for q in quality_scores if q >= 20)
    q30_count = sum(1 for q in quality_scores if q >= 30)

    print(f"\nBases >= Q20 (99% accuracy): {q20_count} ({q20_count/len(quality_scores)*100:.1f}%)")
    print(f"Bases >= Q30 (99.9% accuracy): {q30_count} ({q30_count/len(quality_scores)*100:.1f}%)")
    print("=" * 50 + "\n")

def main():
    # Open webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("=" * 60)
    print("PHRED+33 Quality Score OCR and Analyzer")
    print("=" * 60)
    print("Instructions:")
    print("  - Position paper with quality scores in front of camera")
    print("  - Press 's' to capture and analyze")
    print("  - Press 'p' to show preprocessing preview")
    print("  - Press 'q' to quit")
    print("=" * 60)

    # Generate PHRED+33 character whitelist for Tesseract
    #phred33_chars = ''.join(chr(i) for i in range(33, 127))

    # Tesseract configuration
    # --oem 3: Use default OCR Engine Mode
    # --psm 6: Assume a single uniform block of text
    # Try --psm 7 for single line, or --psm 13 for raw line if psm 6 doesn't work well
    custom_config = f"--oem 3 --psm 7" #-c tessedit_char_whitelist={phred33_chars}"

    preview_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the live frame or preprocessed preview
        if preview_mode:
            processed = preprocess_for_ocr(frame)
            # Convert back to BGR for display
            display_frame = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
            cv.imshow('Preprocessing Preview (Press p to toggle)', display_frame)
        else:
            cv.imshow('Webcam Feed (Press s to capture, p for preview)', frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('p'):
            preview_mode = not preview_mode
            cv.destroyAllWindows()

        elif key == ord('s'):
            print("\n" + "-" * 60)
            print("Capturing and processing...")
            print("-" * 60)

            # Save original screenshot
            screenshot_path = 'screenshot.png'
            cv.imwrite(screenshot_path, frame)
            print(f"✓ Original screenshot saved as '{screenshot_path}'")

            # Preprocess the image
            processed = preprocess_for_ocr(frame)
            processed_path = 'processed.png'
            cv.imwrite(processed_path, processed)
            print(f"✓ Preprocessed image saved as '{processed_path}'")

            # Perform OCR using PyTesseract
            try:
                pil_image = Image.fromarray(processed)

                # Extract text with custom configuration
                raw_text = pytesseract.image_to_string(pil_image, config=custom_config)

                print("\nRaw OCR Output:")
                print("-" * 60)
                print(repr(raw_text))  # Use repr to see whitespace/special chars
                print("-" * 60)

                # Validate and clean the quality string
                quality_string = validate_phred33(raw_text)

                if quality_string:
                    print(f"\nCleaned Quality String ({len(quality_string)} characters):")
                    print("-" * 60)
                    print(quality_string)
                    print("-" * 60)

                    # Convert to quality scores
                    quality_scores = phred33_to_quality_scores(quality_string)

                    print("\nQuality Scores (first 50):")
                    print("-" * 60)
                    print(quality_scores[:50])
                    if len(quality_scores) > 50:
                        print(f"... and {len(quality_scores) - 50} more")
                    print("-" * 60)

                    # Print statistics
                    print_quality_statistics(quality_scores)

                    # Generate and save graph
                    print("Generating quality score graph...")
                    plot_quality_scores(quality_scores, quality_string)
                    print("✓ Analysis complete!")

                else:
                    print("\n⚠ Warning: No valid PHRED+33 characters detected!")
                    print("Tips:")
                    print("  - Ensure the text is clearly visible and in focus")
                    print("  - Try better lighting conditions")
                    print("  - Check that the paper is flat and not at an angle")
                    print("  - Press 'p' to see preprocessing preview")

            except Exception as e:
                print(f"\n✗ OCR failed: {e}")
                import traceback
                traceback.print_exc()

            print("-" * 60 + "\n")

    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    print("\nProgram terminated. Goodbye!")

if __name__ == "__main__":
    main()
