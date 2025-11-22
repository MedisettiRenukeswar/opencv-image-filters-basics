import cv2
import os

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    print("Current working directory:", os.getcwd())

    input_path = os.path.join("samples", "input.jpg")
    print("Trying to load image from:", input_path)
    print("File exists:", os.path.exists(input_path))

    output_dir = "outputs"
    ensure_dir(output_dir)

    img = cv2.imread(input_path)
    if img is None:
        print("❌ Failed to load image. Check path / file name / extension.")
        return
    else:
        print("✅ Image loaded successfully. Shape:", img.shape)

    # 0. Original
    cv2.imwrite(os.path.join(output_dir, "0_original.jpg"), img)
    print("Saved 0_original.jpg")

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "1_gray.jpg"), gray)
    print("Saved 1_gray.jpg")

    # 2. Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, "2_blur.jpg"), blur)
    print("Saved 2_blur.jpg")

    # 3. Canny Edge Detection
    edges = cv2.Canny(blur, 100, 200)
    cv2.imwrite(os.path.join(output_dir, "3_edges.jpg"), edges)
    print("Saved 3_edges.jpg")

    # 4. Binary Threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, "4_threshold.jpg"), thresh)
    print("Saved 4_threshold.jpg")

    print(f"✅ All processed images saved in: {output_dir}")

if __name__ == "__main__":
    main()
