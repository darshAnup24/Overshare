import torch
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
from easyocr import Reader
from PIL import Image
import os
from groq import Groq
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse


app=FastAPI()
# Ensure the API key is available
api_key = os.environ.get("GROQ_API_KEY", "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")  # Replace with actual API key
client = Groq(api_key=api_key)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize DETR model on GPU if available
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

# Initialize EasyOCR reader
reader = Reader(['en'], gpu=torch.cuda.is_available())

# Ensure the 'temp' directory exists
if not os.path.exists('temp'):
    os.makedirs('temp')

# Utility function: Perform OCR using EasyOCR
def perform_ocr(image_path):
    try:
        results = reader.readtext(image_path)
        extracted_text = " ".join([res[1] for res in results])
        return extracted_text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return "Error: OCR processing failed"

def blur_text_in_image(image_path):
    """
    Blurs detected text in an image using EasyOCR and OpenCV.
    
    Parameters:
        image_path (str): Path to the input image.
        
    Returns:
        blurred_image (numpy.ndarray): Image with blurred text regions.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Initialize EasyOCR reader
    reader = Reader(['en'])

    # Detect text
    results = reader.readtext(image)

    # Iterate through detected text boxes
    for (bbox, text, prob) in results:
        # Get the bounding box coordinates
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x_min = int(min(top_left[0], bottom_left[0]))
        y_min = int(min(top_left[1], top_right[1]))
        x_max = int(max(bottom_right[0], top_right[0]))
        y_max = int(max(bottom_right[1], bottom_left[1]))

        # Extract the region of interest (ROI)
        roi = image[y_min:y_max, x_min:x_max]

        # Apply Gaussian blur to the ROI
        blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

        # Replace the original ROI with the blurred ROI
        image[y_min:y_max, x_min:x_max] = blurred_roi

    return image

# Utility function: Perform object detection using DETR
def perform_object_detection(image_path):
    try:
        image = Image.open(image_path)
        inputs = detr_processor(images=image, return_tensors="pt").to(device)
        outputs = detr_model(**inputs)
        results = detr_processor.post_process_object_detection(outputs, target_sizes=[image.size])[0]
        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.8:  # Confidence threshold
                detected_objects.append({
                    "label": detr_model.config.id2label[label.item()],
                    "box": box.tolist(),
                    "confidence": score.item()
                })
        return detected_objects
    except Exception as e:
        print(f"Error during object detection: {e}")
        return []

# Utility function: Generate detailed description using Groq API
def generate_detailed_description(text, objects):
    try:
        prompt = (
            f"Analyze the following image and provide the following details:\n\n"
            f"Text Extracted: {text}\n\n"
            f"Objects Detected: {objects}\n\n"
            f"1. **Image Description:**\n"
            f"   - Provide a concise and detailed textual description of the image contents based on the text and objects detected.\n\n"
            f"2. **Sensitive Information Risks:**\n"
            f"   - Identify any sensitive information such as phone numbers, email addresses, or personal IDs. "
            f"   Provide details on whether any is found and its severity.\n\n"
            f"3. **NSFW Content Risks:**\n"
            f"   - Check for any inappropriate or explicit content. Flag any detected NSFW content and rate the risk level.\n\n"
            f"4. **Object Detection Risks:**\n"
            f"   - List any detected objects that could pose a risk based on context (e.g., confidential documents, weapons, etc.). "
            f"   Provide a severity assessment for each.\n\n"
            f"5. **Confidence Levels:**\n"
            f"   - Include confidence levels for all detections (text, objects, sensitive information, and NSFW content).\n\n"
            f"Provide a structured report detailing any findings and suggested actions for mitigating risks."
        )

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error during API request: {e}")
        return {"error": str(e)}

# Main function: Process image and generate output
# Update the process_image function
def process_image(image_path):
    try:
        # Step 1: Blur detected text in the image
        blurred_image = blur_text_in_image(image_path)
        blurred_image_path = image_path.replace(".jpg", "_blurred.jpg")
        cv2.imwrite(blurred_image_path, blurred_image)

        # Step 2: Extract text using OCR
        extracted_text = perform_ocr(image_path)
        print(f"Extracted Text: {extracted_text}")

        # Step 3: Detect objects using DETR
        detected_objects = perform_object_detection(image_path)
        print(f"Detected Objects: {detected_objects}")

        # Step 4: Generate detailed description with Groq API
        detailed_analysis = generate_detailed_description(extracted_text, detected_objects)
        print(f"Detailed Analysis: {detailed_analysis}")

        return {
            "Blurred Image Path": blurred_image_path,
            "Extracted Metadata": extracted_text,
            "Detected Objects": detected_objects,
            "Detailed Risk Analysis": detailed_analysis
        }
    except Exception as e:
        print(f"Error during image processing: {e}")
        return {"error": f"Image processing failed: {str(e)}"}


# Update the analyze_image endpoint
@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"temp/{file.filename}"
        print(f"File uploaded to: {file_location}")
        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())
        
        # Process the image using your models
        output = process_image(file_location)
        
        # Prepare the blurred image for display
        blurred_image_path = output.get("Blurred Image Path")
        blurred_image_url = f"/temp/{os.path.basename(blurred_image_path)}"

        # Clean up the original image
        os.remove(file_location)
        
        return {
            "Blurred Image URL": blurred_image_url,
            "Extracted Metadata": output.get("Extracted Metadata"),
            "Detected Objects": output.get("Detected Objects"),
            "Detailed Risk Analysis": output.get("Detailed Risk Analysis"),
        }
    except Exception as e:
        print(f"Error during analysis: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


# Serve static files for temp directory
from fastapi.staticfiles import StaticFiles
app.mount("/temp", StaticFiles(directory="temp"), name="temp")


# Update HTML response to display blurred image
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Analysis</title>
    </head>
    <body>
        <h1>Upload an Image for Analysis</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="imageFile" name="file" accept="image/*" required>
            <button type="submit">Upload</button>
        </form>
        <div id="result"></div>

        <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData();
            formData.append('file', document.getElementById('imageFile').files[0]);

            const response = await fetch('/analyze/', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            const blurredImageUrl = data['Blurred Image URL'] || '';
            document.getElementById('result').innerHTML = `
                <h2>Results</h2>
                <p><strong>Extracted Metadata:</strong> ${data['Extracted Metadata']}</p>
                <p><strong>Detected Objects:</strong> ${JSON.stringify(data['Detected Objects'])}</p>
                <p><strong>Risk Analysis:</strong> ${data['Detailed Risk Analysis']}</p>
                <p><strong>Blurred Image:</strong></p>
                <img src="${blurredImageUrl}" alt="Blurred Image" style="max-width:100%;"/>
            `;
        });
        </script>
    </body>
    </html>
    """
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
