# main.py
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from utils.predict import predict_image

app = FastAPI()
# INSERT THIS RIGHT BELOW
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can use specific domains like ["http://localhost:3000"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Brain Tumor Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    predicted_label, probabilities = predict_image(image)
    
    label_map = {
        0: "Glioma",
        1: "Meningioma",
        2: "No Tumor",
        3: "Pituitary"
    }

    predicted_class = label_map.get(predicted_label, "Unknown")

    return {
            "prediction": predicted_class,
            "confidence": round(float(probabilities[predicted_label]) * 100, 1),
            "scores": {
                "Glioma": round(float(probabilities[0]) * 100, 1),
                "Meningioma": round(float(probabilities[1]) * 100, 1),
                "No Tumor": round(float(probabilities[2]) * 100, 1),
                "Pituitary": round(float(probabilities[3]) * 100, 1),
            }
        }