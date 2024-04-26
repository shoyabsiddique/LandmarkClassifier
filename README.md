# Landmark Image Classifier API

This repository contains code for deploying a landmark image classifier model as a FastAPI-based API. The model predicts the landmark depicted in an uploaded image and provides additional information about the landmark.

## Features

- **Image Classification**: Classifies uploaded images into landmark categories.
- **Landmark Identification**: Utilizes Google Generative AI to identify landmarks in images.
- **Detailed Summary**: Generates detailed summaries for identified landmarks.
- **Nearby Landmarks**: Provides suggestions for nearby landmarks based on the identified landmark.

## Prerequisites

Before running the API, make sure you have the following installed:

- Python 3.x
- Pip (Python package manager)

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/landmark-image-classifier-api.git
```

2. Navigate to the project directory:

```bash
cd landmark-image-classifier-api
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:

```bash
uvicorn main:app --reload
```

2. Access the API documentation in your web browser at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to upload an image and classify it.

## Example

Here's a simple example of how to use the API:

1. Upload an image of a landmark through the API documentation interface.
2. Receive a JSON response containing the predicted landmark name, confidence score, detailed information about the landmark, and nearby landmark suggestions.

## Credits

- **Google Generative AI**: Used for landmark identification and detailed summary generation.
- **PyTorch**: Used for training and deploying the landmark image classifier model.
- **FastAPI**: Used for building the RESTful API.
- **FuzzyWuzzy**: Used for fuzzy matching in identifying nearby landmarks.
