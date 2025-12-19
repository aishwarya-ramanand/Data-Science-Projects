# 🎙️ GenAI Voice-Based Interview Assessment System

**By Aishwarya R**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![GenAI](https://img.shields.io/badge/GenAI-Gemini-orange)
![Speech](https://img.shields.io/badge/Speech-STT-green)
![Analytics](https://img.shields.io/badge/Analytics-Google_Sheets-yellow)

---

## 🧠 Overview

This project is an **end-to-end GenAI-powered voice-based interview assessment system** designed to simulate real interview scenarios.

The system dynamically generates interview questions, captures spoken responses, transcribes them using speech-to-text, evaluates answers using multiple scoring rubrics, and logs structured analytics for performance tracking.

The focus of this project is on **LLM orchestration, speech processing, evaluation logic, and data logging**, making it highly relevant for **AI, Data, and ML-focused roles**.

---

## ✨ Key Features

- 🎯 **Dynamic Interview Question Generation** using Google Gemini
- 🎙️ **Voice Answer Capture** via microphone input
- 📝 **Speech-to-Text Transcription** using OpenAI Whisper
- 📊 **Multi-Rubric Answer Evaluation**, including:
  - Clarity
  - Accuracy
  - Completeness
  - Fluency
  - Keyword relevance
- 🔁 **Adaptive Question Regeneration** based on candidate performance
- 📈 **Real-Time Logging & Analytics** using Google Sheets
- 🖥️ **Interactive UI** built with Streamlit

---

## 🏗️ Project Architecture (High-Level)

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/7d072fc3-309e-4a15-b173-37d456be7410" />


---

## 🧰 Tech Stack

- **Programming Language:** Python
- **LLM:** Google Gemini
- **Speech-to-Text:** OpenAI Whisper
- **UI Framework:** Streamlit
- **Data Logging:** Google Sheets API
- **Audio Handling:** SoundDevice, SciPy
- **Evaluation Logic:** Prompt-based LLM scoring

---

## 📊 Evaluation Strategy

Each spoken answer is evaluated using **multiple qualitative and quantitative rubrics**, enabling a more realistic interview assessment than simple correctness checks.

Scores are stored in structured format to support:
- Candidate performance tracking
- Trend analysis
- Automated reporting

---

## 🚧 About AI Calling (Telephony)

An AI calling (telephony) extension using Asterisk was explored during development.  
However, the **core assessment system is fully functional without telephony**, and the project was intentionally finalized at this stage to maintain stability and clarity.

> If asked:  
> *“The AI calling component was an experimental extension. The finalized project focuses on robust voice-based assessment through browser and local audio pipelines.”*

---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/genai-voice-interview.git
cd genai-voice-interview

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py

📌 Key Learnings
	•	Practical integration of LLMs with speech pipelines
	•	Designing evaluation systems beyond accuracy
	•	Real-world challenges of GenAI orchestration
	•	Logging AI outputs for analytics and reporting
	•	Building interview-ready AI systems

👩‍💻 Author

Aishwarya R
Aspiring Data Scientist | GenAI & ML Enthusiast

---

### ✅ Next (recommended)
If you want, I can:
- Rewrite this **for ATS-style GitHub recruiters**
- Create a **one-paragraph GitHub description**
- Convert this into **resume bullet points**
- Help you answer interview questions confidently

Just tell me 💙
