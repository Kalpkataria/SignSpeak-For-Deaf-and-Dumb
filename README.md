# SignSpeak – AI-Based Assistive Communication System

SignSpeak is an AI-powered assistive communication platform designed to help people with hearing and speech impairments communicate more effectively.  
The system converts multiple types of user inputs such as text, images, and hand gestures into sign language representations using artificial intelligence.

---

## Features

- Text to Sign Language Processing
- Image to Text Extraction using OCR
- Hand Sign Recognition using Deep Learning
- Multi-modal Input Support
- AI-based Natural Language Processing
- Simple and User-Friendly Interface

---

## Technologies Used

### Frontend
- React
- JavaScript
- Vite
- HTML / CSS

### Backend
- FastAPI
- Python

### AI & Machine Learning
- TensorFlow (CNN model for hand sign recognition)
- spaCy (Natural Language Processing)
- OpenCV (Image Processing)
- Tesseract OCR (Text extraction from images)

---

## System Architecture

```
Frontend (React)
      │
      │ API Requests
      ▼
Backend (FastAPI)
      │
      ├── Text Processing (spaCy NLP)
      │
      ├── OCR Processing (OpenCV + Tesseract)
      │
      └── Hand Sign Recognition (TensorFlow CNN)
```

---

## How It Works

1. User provides input such as text, speech, or image.
2. Frontend sends the input to the backend through API requests.
3. Backend processes the data using AI models.
4. The processed output is converted into sign language representations.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/signspeak.git
cd signspeak
```

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## Future Improvements

- Real-time hand gesture recognition using webcam
- Speech to sign language conversion
- Mobile application support
- Improved AI model accuracy

---

## Purpose

The goal of this project is to reduce communication barriers and make communication easier for people with hearing and speech impairments using artificial intelligence.
