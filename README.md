# MNIST Digit Recognizer CNN Visualization Web App

This web application allows you to visualize how a CNN (Convolutional Neural Network) processes MNIST digit images. You can upload an image of a handwritten digit, and the app will show you the feature maps at each layer of the CNN and provide the final prediction.

## Project Structure

```
├── backend/             # FastAPI backend
│   ├── main.py         # Main FastAPI application
│   └── requirements.txt # Python dependencies
├── frontend/           # HTML/CSS/JS frontend
│   ├── index.html      # Main HTML page
│   ├── styles.css      # CSS styles
│   ├── script.js       # JavaScript for frontend functionality
│   └── upload-icon.svg # Upload icon
└── ml_model/           # Machine learning model
    └── models/         # Trained CNN models
```

## Features

- Upload images of handwritten digits
- Visualize feature maps from each convolutional and pooling layer
- See the model's prediction and confidence level
- Interactive probability visualization for all digit classes

## How to Run

### Backend Setup

1. Install the required Python packages:
   ```
   cd backend
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:
   ```
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. Open the frontend/index.html file in a web browser, or serve it using a simple HTTP server:
   ```
   cd frontend
   python -m http.server 8080
   ```

2. Access the application at http://localhost:8080

## How It Works

1. The frontend allows you to upload an image of a handwritten digit
2. The image is sent to the FastAPI backend
3. The backend processes the image through the CNN model
4. The model extracts feature maps at each layer
5. The backend returns the feature maps and prediction
6. The frontend displays the results visually

## CNN Architecture

The CNN model has the following architecture:

1. Input: 28x28 grayscale image
2. First Convolutional Layer: 16 filters of size 3x3
3. ReLU Activation
4. Max Pooling: 2x2
5. Second Convolutional Layer: 32 filters of size 3x3
6. ReLU Activation
7. Max Pooling: 2x2
8. Flatten
9. Fully Connected Layer: 128 neurons
10. ReLU Activation
11. Output Layer: 10 neurons (one for each digit 0-9)
