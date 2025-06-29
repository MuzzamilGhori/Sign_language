from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
def predict_sign_from_image(img): ...
def get_sign_image_paths(words): ...

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Sign Language Interpreter API"}

@app.post("/predict-sign/")
async def predict_sign(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    prediction = predict_sign_from_image(img)
    return JSONResponse({"prediction": prediction})

@app.get("/text-to-sign/")
def text_to_sign(text: str):
    words = text.upper().split()
    image_data = get_sign_image_paths(words)
    return JSONResponse({"images": image_data})
