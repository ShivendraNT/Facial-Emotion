import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import os
from collections import deque

import torch.nn as nn

device='mps'
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu=nn.ReLU()
        self.batchnorm1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.batchnorm2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.batchnorm3=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.batchnorm4=nn.BatchNorm2d(256)
        self.linear1=nn.Linear(in_features=2304,out_features=512)
        self.dropout=nn.Dropout(0.5)
        self.linear2=nn.Linear(in_features=512,out_features=7)
        
    def forward(self,x):
        y=self.conv1(x)
        y=self.batchnorm1(y)
        y=self.relu(y)
        y=self.maxpool(y)
        y=self.conv2(y)
        y=self.batchnorm2(y)
        y=self.relu(y)
        y=self.maxpool(y)
        y=self.conv3(y)
        y=self.batchnorm3(y)
        y=self.relu(y)
        y=self.maxpool(y)
        y=self.conv4(y)
        y=self.batchnorm4(y)
        y=self.relu(y)
        y=self.maxpool(y)
        y=torch.flatten(y,1)
        y=self.linear1(y)
        y=self.relu(y)
        y=self.dropout(y)
        y=self.linear2(y)

        return y
    
import cv2


model=EmotionCNN().to('mps')
model.load_state_dict(torch.load("emotion_model.pth",map_location='mps'))
model.eval()


# Load the pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_map = {
    0: ("Angry", ">:("),
    1: ("Disgust", ":-X"),
    2: ("Fear", ":-O"),
    3: ("Happy", ":)"),
    4: ("Neutral", ":|"),
    5: ("Sad", ":("),
    6: ("Surprise", ":O")
}



def preprocess_face(gray, coords):
    x, y, w, h = coords
    padding = 10
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, gray.shape[1])
    y2 = min(y + h + padding, gray.shape[0])

    face_crop = gray[y1:y2, x1:x2]
    face_crop = cv2.equalizeHist(face_crop)  # <--- improves lighting balance
    resized = cv2.resize(face_crop, (48, 48))
    normalized = (resized / 255.0 - 0.5) / 0.5
    tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0).float().to(device)
    return tensor

def predict_emotion(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
        prob = torch.softmax(output, dim=1)[0][pred].item()
        recent_preds.append(pred)
        final_pred = max(set(recent_preds), key=recent_preds.count)

    return final_pred, prob

cap=cv2.VideoCapture(0)

frame_count = 0
frame_skip=10
recent_preds = deque(maxlen=5)
last_pred, last_prob = 6, 1  # start with "neutral"

# Loop

while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:

        frame_count += 1

        # Run inference every N frames
        if frame_count % frame_skip == 0:
            tensor = preprocess_face(gray, (x, y, w, h))
            pred, prob = predict_emotion(model, tensor)
            recent_preds.append(pred)
            last_pred = max(set(recent_preds), key=recent_preds.count)  # majority vote
            last_prob = prob

        # Get label + emoji
        label, emoji = emotion_map[last_pred]

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emoji} {label} ({last_prob*100:.1f}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Show the frame (even if no faces detected)
    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()