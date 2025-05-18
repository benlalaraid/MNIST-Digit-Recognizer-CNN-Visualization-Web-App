import os
import io
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the CNN model architecture (same as in the notebook)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # (1,28,28) -> (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # -> (16,14,14)
            nn.Conv2d(16, 32, 3, padding=1), # -> (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # -> (32,7,7)
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        # Get intermediate outputs for visualization
        conv1 = self.network[0](x)
        relu1 = self.network[1](conv1)
        pool1 = self.network[2](relu1)
        
        conv2 = self.network[3](pool1)
        relu2 = self.network[4](conv2)
        pool2 = self.network[5](relu2)
        
        # For the final output, run through the entire network
        output = self.network(x)
        
        return output, conv1, pool1, conv2, pool2

# Load the model
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "ml_model", "models", "simple_cnn_state_dict.pth")

device = torch.device("cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def tensor_to_base64(tensor):
    """Convert a tensor to a base64 encoded image"""
    # Normalize to 0-1 range
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    # Convert to numpy array and then to PIL Image
    img = Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))
    # Save to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    # Convert to base64
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")  # Convert to grayscale
        
        # Transform the image
        tensor_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        
        # Get the original image as base64
        original_image_base64 = base64.b64encode(contents).decode("utf-8")
        
        # Forward pass with visualization
        with torch.no_grad():
            logits, conv1_output, pool1_output, conv2_output, pool2_output = model(tensor_image)
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # Convert feature maps to base64 images
        conv1_visualizations = []
        for i in range(conv1_output.size(1)):
            feature_map = conv1_output[0, i]
            conv1_visualizations.append(tensor_to_base64(feature_map))
        
        pool1_visualizations = []
        for i in range(pool1_output.size(1)):
            feature_map = pool1_output[0, i]
            pool1_visualizations.append(tensor_to_base64(feature_map))
        
        conv2_visualizations = []
        for i in range(conv2_output.size(1)):
            feature_map = conv2_output[0, i]
            conv2_visualizations.append(tensor_to_base64(feature_map))
        
        pool2_visualizations = []
        for i in range(pool2_output.size(1)):
            feature_map = pool2_output[0, i]
            pool2_visualizations.append(tensor_to_base64(feature_map))
        
        # Return the results
        return {
            "original_image": original_image_base64,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "conv1_output": conv1_visualizations,
            "pool1_output": pool1_visualizations,
            "conv2_output": conv2_visualizations,
            "pool2_output": pool2_visualizations,
            "probabilities": probabilities.tolist()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "MNIST Digit Recognizer API"}
