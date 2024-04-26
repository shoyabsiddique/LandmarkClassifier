# Import necessary libraries
import numpy as np
from typing import Annotated
from fastapi import FastAPI, File
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import torchvision.transforms as T
from PIL import Image
import io
import numpy as np
import uvicorn
import google.generativeai as genai
from fuzzywuzzy import process


# Initialize FastAPI app
app = FastAPI()

# Load the model
model_path = "unf_layer4_landmark_classifier.pt"
model = torch.jit.load(model_path)
model.eval()

# Gemini API key
GOOGLE_API_KEY = ""

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')
gemini_vision_model = genai.GenerativeModel('gemini-pro-vision')

# List of valid landmarks
valid_landmarks = [
    'Kala Ghoda',
    'Vivekanand Education Society_s Institute of Technology (VESIT) - Chembur',
    'One Indiabulls Centre',
    'Haji Ali Dargah, Worli',
    'The Oberoi, Mumbai',
    'Babulnath Temple',
    'Wadiaji Atash Behram, Grant Road',
    'Imperial Towers',
    'St. Xavier_s College - Fort',
    'Oberoi Garden City',
    'Sanjay Gandhi National Park',
    'Gloria Church, Byculla',
    'Raheja Towers',
    'Wilson College - Chowpatty',
    'Rizvi College of Engineering - Bandra',
    'Teerthdham Mangalayatan, Vasai',
    'Rajiv Gandhi Institute of Technology (RGIT) - Versova',
    'Powai Lake',
    'Mahalakshmi Temple, Mahalaxmi',
    'Sion Fort',
    'Grant Medical College - Byculla',
    'Taraporewala Aquarium',
    'Sewri Fort',
    'Bandra-Worli Sea Link',
    'Reserve Bank of India (RBI) building',
    'The Leela Mumbai',
    'Gilbert Hill',
    'The Asiatic Society of Mumbai',
    'Magen David Synagogue, Byculla',
    'National Institute of Fashion Technology (NIFT) Mumbai - Kharghar',
    'Pillai College of Engineering - Panvel',
    'Indian Institute of Technology (IIT) Bombay - Powai',
    'Rustomjee Business School - Dahisar',
    'Sophia College for Women - Peddar Road',
    'SNDT Women_s University - Churchgate',
    'Khada Parsi Statue',
    'Siddhivinayak Temple, Prabhadevi',
    'Sir J.J. College of Architecture - Fort',
    'Bhau Daji Lad Museum',
    'Antilia',
    'Ghodbunder Fort',
    'Veermata Jijabai Technological Institute (VJTI) - Matunga',
    'Taj Mahal Palace Hotel',
    'Afghan Church',
    'M.H. Saboo Siddik College of Engineering - Byculla',
    'Rajabai Clock Tower',
    'Vasai Fort',
    'Bombay Art Society',
    'Kanheri Caves',
    'Nehru Centre',
    'Flora Fountain',
    'Bombay Stock Exchange (BSE)',
    'St. Michael_s Church, Mahim',
    'Chhatrapati Shivaji Maharaj Vastu Sangrahalaya',
    'Walkeshwar Temple, Walkeshwar',
    'nirlon knowledge park',
    'royal opera house mumbai',
    'ISKCON Temple, Juhu',
    'Taj Lands End',
    'Jama_Masjid',
    'Eliphistone college',
    'Marine Drive',
    'Tata Institute of Social Sciences (TISS) - Deonar',
    'Sardar Patel Institute of Technology (SPIT) - Andheri West',
    'Jai Hind College - Churchgate',
    'SIES Graduate School of Technology - Nerul',
    'Banganga Tank',
    'Regal Cinema',
    'Narsee Monjee Institute of Management Studies - VIle Parle',
    'Renaissance Mumbai Convention Centre Hotel',
    'Gateway of India',
    'David Sassoon Library and Reading Room',
    'Mahalaxmi Racecourse',
    'Worli Sea Face',
    'Mani Bhavan Gandhi Museum',
    'Victoria Terminus (Chhatrapati Shivaji Maharaj Terminus)',
    'Hanging Gardens',
    'University of Mumbai - Kalina, Santacruz',
    'Worli Fort',
    'Jamnalal_Bajaj_Institute_of_Management_Studies_(JBIMS)',
    'Global Vipassana Pagoda, Gorai',
    'S.P. Jain Institute of Management and Research (SPJIMR) - Andheri West',
    'Mumba Devi Temple',
    'Lokmanya Tilak Terminus',
    'K. J. Somaiya College of Engineering - Vidyavihar',
    'Peninsula Corporate Park',
    'Bombay High Court',
    'St. Thomas Cathedral'
]

# Function to get landmark name from Gemini API
def get_landmark_name(image):
    try:
        # response = gemini_vision_model.generate_content(["Identify the landmark in this image. If cannot identify return Landmark not recognized", image], stream=True)
        response = gemini_vision_model.generate_content(["Identify and provide just the name of the landmark in this image or 'Not recognized' if none found", image], stream=True)

        response.resolve()  # Wait for the response to complete
        landmark_name = response.text.strip()  # Extract the generated text, removing potential whitespace
        return landmark_name
    
    except Exception as e:  # Catch potential errors gracefully
        print(f"Error identifying landmark: {e}")
        return "An error occurred while identifying the landmark. Please try again later."
        

# Function to get detailed summary from Gemini API
def get_detailed_summary(landmark):
    try:
        prompt = f"""
        **Landmark:** {landmark}

        **Location:** (Optional: Add location information if available)

        **History:** Briefly describe {landmark}'s history and construction based on reliable sources.

        **Significance:** Explain the cultural or historical significance of {landmark}.

        **Architecture:** Describe the architectural style and notable features of {landmark}.

        **Interesting Facts:** Share any 3 interesting facts or trivia about {landmark}.
        """

        response = gemini_model.generate_content(prompt)
        summary = response.text.strip()  # Extract the generated text, removing potential whitespace
        return summary
    except Exception as e:  # Catch potential errors gracefully
        print(f"Error generating summary: {e}")
        return "An error occurred while generating the summary. Please try again later."

# Function to get nearby landmark suggestions from the valid list
def get_nearby_landmarks(landmark_name):
    try:
        # Create a prompt that asks for nearby landmarks from the valid list
        valid_landmarks_str = ', '.join(valid_landmarks)
        prompt = f"Can you suggest 5 nearest landmarks to {landmark_name} from this list: {valid_landmarks_str} with distance between them."

        response = gemini_model.generate_content(prompt)
        suggestions = response.text.strip().split('\n')  # Extract the generated suggestions as a list

        # Filter out suggestions that are not in the valid list of landmarks
        nearby_landmarks = []
        for suggestion in suggestions:
            # Use fuzzy matching to find the closest matches in the valid landmarks list
            matches = process.extract(suggestion, valid_landmarks, limit=3)

            # Only add the matches if the score is above a certain threshold (e.g., 90)
            for match, score in matches:
                if score > 90:
                    nearby_landmarks.append(match)

        return nearby_landmarks
    except Exception as e:  # Catch potential errors gracefully
        print(f"Error generating nearby landmarks: {e}")
        return []

# Prediction function
def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    timg = T.ToTensor()(img).unsqueeze_(0)

    with torch.no_grad():
        outputs = model(timg)
        # If your model already applies softmax, you can comment out the next line
        # outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs_np = outputs.cpu().numpy().squeeze()

    top5_idx = np.argsort(outputs_np)[-5:][::-1]
    top5_values = outputs_np[top5_idx]
    top5_classes = [model.class_names[i] for i in top5_idx]

    return top5_classes, top5_values

@app.get("/", response_class=HTMLResponse)
def upload_page():
    return """
    <!doctype html>
    <html>
        <head>
            <title>Upload an Image</title>
        </head>
        <body>
            <h1>Upload an Image to Classify</h1>
            <form action="/classify/" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

# Endpoint to classify image
@app.post("/classify/")
async def classify_image(file: Annotated[bytes, File()]):
    # image_bytes = await file.read()
    # print(str(file))
    top5_classes, top5_confidences = predict_image(file)

    # Get the first prediction
    first_prediction = top5_classes[0]
    confidence = top5_confidences[0]

    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(file))

    # Get the landmark name from Gemini API
    landmark_name = get_landmark_name(image)

    # Choose between landmark_name and first_prediction based on whether landmark_name is recognized
    match, score = process.extractOne(landmark_name, top5_classes)
    if score > 90 and landmark_name != 'Not recognized':
        summary_landmark = match
    elif top5_confidences[0] > 0.7 and landmark_name != 'Not recognized':
        summary_landmark = first_prediction
    elif top5_confidences[0] > 0.35 and landmark_name != 'Not recognized':
        summary_landmark = "maybe " + first_prediction
    else:
        match, score = process.extractOne(landmark_name, valid_landmarks)
        if score > 80 and landmark_name != 'Not recognized':
            summary_landmark = "maybe " + landmark_name
            confidence = 0.6500
        else:
            summary_landmark = "Sorry, unable to identify landmark"

    if summary_landmark != "Sorry, unable to identify landmark":
        detailed_summary = get_detailed_summary(summary_landmark)
        nearby_landmarks = get_nearby_landmarks(summary_landmark)
    else:
        detailed_summary = None
        nearby_landmarks = None
        confidence = 0
    
    response_data = {
    "Landmark Name": summary_landmark,
    "Confidence Score": f"{float(confidence) * 100:.2f}%",
    "Landmark Information": detailed_summary,
    "Nearby Landmarks": nearby_landmarks,
}
    return JSONResponse(content=response_data)

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)