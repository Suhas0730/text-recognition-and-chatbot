import easyocr
import cv2
import os
from config import *

# Initialize the EasyOCR reader with the handwritten language model
reader = easyocr.Reader(['en', 'en-handwritten'])  # Adding 'en-handwritten' for better handwriting recognition

# Get the current working directory
cwd = os.getcwd()

# Get the path to the input folder and list all the image files
images_path = os.path.join(cwd, input_folder)
image_names = os.listdir(images_path)

# Create the output directory if it doesn't exist
output_dir = os.path.join(cwd, preprocessed_directory)
os.makedirs(output_dir, exist_ok=True)

for image_name in image_names:
    # Get the path to the image
    image_path = os.path.join(images_path, image_name)
    
    # Load the image and convert it to grayscale
    img = cv2.imread(image_path)
    
    # Apply preprocessing for handwritten text:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to enhance text contrast
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Optionally, apply dilation or erosion to clean the text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Perform OCR on the preprocessed image
    result = reader.readtext(dilated)

    # Draw bounding boxes and texts on the image
    for res in result:
        if res[2] < min_confidence:  # You can adjust this threshold for confidence
            continue
        start_x, start_y = int(res[0][0][0]), int(res[0][0][1])
        end_x, end_y = int(res[0][2][0]), int(res[0][2][1])
        print(f" Text detected: {res[1]}, confidence: {res[2]:.3f}")
        
        # Draw a rectangle around detected text
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(img, res[1], (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Save the output image with bounding boxes
    output_path = os.path.join(output_dir, image_name)
    print(f"[INFO] saved output to {output_path}")
    cv2.imwrite(output_path, img)
    print("------------------------------------")
    print("")
