# test1.py
# Final LLM Voice Agent: multi-topic 5Q each, spoken-answer-friendly, Sheets logging
#
# NOTE: This script includes your GEMINI_API_KEY inline as requested (Option A).
#       Storing API keys in code is insecure; consider using env vars later.

from kokoro import KPipeline
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import time
import numpy as np
import soundfile as sf
import os
import google.generativeai as genai
import json
import traceback

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# ----------------- CONFIG ----------------- #
GEMINI_API_KEY = "GEMINI_API_KEY"   # <-- your key (as requested)
SHEET_ID = "SHEET_NAME"  # your sheet id
MODEL_NAME = "models/gemini-2.5-flash"   # model to use
TOPICS = ["SQL", "Python", "Statistics", "Machine Learning"]
QUESTIONS_PER_TOPIC = 5
AUDIO_SR = 16000

# ----------------- INITIALIZE ----------------- #
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name=MODEL_NAME)

pipeline = KPipeline(lang_code="a")      # Kokoro TTS
stt_model = whisper.load_model("base")   # Whisper STT

# ----------------- Google Sheets ----------------- #
def ensure_sheet_header(sheet):
    try:
        first_row = sheet.row_values(1)
        if not first_row:
            header = [
                "Timestamp","Topic","Question No","Question","Student Answer",
                "Communication Clarity","Technical Accuracy","Keyword Usage",
                "Confidence Fluency","Completeness","Total Score","Correctness",
                "Feedback","Improvement","Difficulty","Topic Tag"
            ]
            sheet.insert_row(header, index=1)
            print("[SHEET] Header row inserted.")
    except Exception as e:
        print("[SHEET HEADER ERROR]", e)
        traceback.print_exc()

def connect_to_sheet():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_file("service_account.json", scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).sheet1
    # ensure header exists
    try:
        ensure_sheet_header(sheet)
    except Exception:
        pass
    return sheet

def save_row_to_sheet(row):
    try:
        sheet = connect_to_sheet()
        sheet.append_row(row)
        print("[SAVED TO SHEET]")
    except Exception as e:
        print("[ERROR SAVING TO SHEET]", e)
        traceback.print_exc()
        # Optionally fallback to local CSV
        try:
            import csv
            with open("assessment_fallback.csv", "a", newline="", encoding="utf-8") as f:
                cw = csv.writer(f)
                cw.writerow(row)
            print("[SAVED TO LOCAL CSV fallback]")
        except Exception as e2:
            print("[ERROR SAVING TO LOCAL CSV]", e2)
            traceback.print_exc()

# ----------------- TTS / Playback ----------------- #
def speak_text(text, filename="question.wav"):
    try:
        generator = pipeline(text, voice="af_heart", speed=1.35, split_pattern=r"\n+")
        audio_chunks = []
        for _, _, audio in generator:
            # generator is expected to yield tuples where audio is a tensor/array
            audio_np = audio.detach().cpu().numpy()
            audio_chunks.append(audio_np)
        if not audio_chunks:
            print("[TTS] No audio generated.")
            return
        final_audio = np.concatenate(audio_chunks)
        sf.write(filename, final_audio, AUDIO_SR)
        print("[TTS] Playing audio...")
        # macOS playback; change if using Linux/Windows
        os.system(f"afplay {filename}")
    except Exception as e:
        print("[TTS ERROR]", e)
        traceback.print_exc()

# ----------------- Recording (ENTER start/ENTER stop) ----------------- #
def record_answer(filename="answer.wav"):
    import threading
    stop_flag = {"stop": False}

    def wait_for_enter():
        input()  # waits for ENTER
        stop_flag["stop"] = True

    # Start the thread
    listener = threading.Thread(target=wait_for_enter)
    listener.daemon = True
    listener.start()

    print("Press ENTER to stop recording...")
    sd.default.samplerate = AUDIO_SR
    sd.default.channels = 1

    audio_frames = []

    def callback(indata, frames, time_, status):
        if stop_flag["stop"]:
            raise sd.CallbackStop()
        audio_frames.append(indata.copy())

    try:
        with sd.InputStream(callback=callback):
            while not stop_flag["stop"]:
                sd.sleep(100)
    except sd.CallbackStop:
        pass
    except Exception as e:
        print("[REC ERROR]", e)

    if len(audio_frames) == 0:
        print("[REC] No audio captured.")
        return None

    audio = np.concatenate(audio_frames, axis=0)
    write(filename, AUDIO_SR, (audio * 32767).astype("int16"))
    print(f"[REC] Saved: {filename}")
    return filename

# ----------------- STT ----------------- #
def speech_to_text(audio_file):
    if audio_file is None:
        return ""
    print("[STT] Transcribing...")
    try:
        result = stt_model.transcribe(audio_file)
        text = result.get("text", "").strip()
    except Exception as e:
        print("[STT ERROR]", e)
        traceback.print_exc()
        text = ""
    print("[STT] Transcript:", text)
    return text

# ----------------- Gemini Helpers ----------------- #
def generate_spoken_questions(topic, n=5):
    prompt = f"""
You are an interviewer creating spoken-answer interview questions for learners.
Generate EXACTLY {n} distinct, concise questions about the topic below.
Important rules:
- Questions must be answerable verbally (explanations, comparisons, descriptions, examples).
- DO NOT ask the learner to "write code" or "write SQL" or otherwise require typing.
- Keep each question short (one sentence).
- Return output as a JSON array of strings, nothing else.

Topic: {topic}
"""
    try:
        resp = gemini_model.generate_content(prompt)
        raw = resp.text.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        qlist = json.loads(cleaned)
        if isinstance(qlist, list) and len(qlist) >= n:
            return qlist[:n]
    except Exception as e:
        print("[GEN QUESTIONS ERROR]", e)
        traceback.print_exc()

    fallback = [f"{topic} question {i+1} — explain or give an example." for i in range(n)]
    return fallback

def relevancy_check_question(question, topic):
    prompt = f"""
You are a short validator. Answer ONLY 'OK' or 'BAD' (one word).
Is the following interview question appropriate for a spoken verbal answer (not requiring code or typed response)?
Question: \"{question}\"
Topic: {topic}
Return 'OK' if it's fine, 'BAD' if it requires writing code or is not answerable verbally.
"""
    try:
        resp = gemini_model.generate_content(prompt)
        txt = resp.text.strip().upper()
        if "OK" in txt:
            return True
    except Exception:
        pass
    return False

def evaluate_answer_with_gemini(question, answer):
    """
    Evaluate the learner's answer across multiple rubrics.
    Expected STRICT JSON response from the model with numeric rubric scores (0-10),
    total_score as the SUM of the five rubric scores (0-50), correctness, feedback,
    improvement, difficulty, and topic_tag.
    """
    eval_prompt = f"""
You are an expert rater. Evaluate the learner's spoken answer to the question below.
Return STRICT JSON only (no commentary). All numeric rubric fields must be integers between 0 and 10.
Compute total_score as the SUM of the five numeric rubric values (so total_score is 0-50).
Return the following JSON keys exactly:

{{
  "communication_clarity": 0-10,
  "technical_accuracy": 0-10,
  "keyword_usage": 0-10,
  "confidence_fluency": 0-10,
  "completeness": 0-10,
  "total_score": 0-50,
  "correctness": "Correct / Incorrect / Partially Correct",
  "feedback": "Short feedback (one or two sentences)",
  "improvement": "One concrete improvement suggestion",
  "difficulty": "Easy / Medium / Hard",
  "topic_tag": "Short tag (e.g., SQL, Python, Regression)"
}}

Question:
{question}

Student Answer:
{answer}

Important: MUST produce valid JSON with the numeric fields as integers, and total_score equal to the sum of the five rubric scores.
"""
    try:
        resp = gemini_model.generate_content(eval_prompt)
        raw = resp.text.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()

        # debug log for dev
        print("[GEMINI RAW OUTPUT]", raw[:1000])  # print up to first 1000 chars

        data = json.loads(cleaned)

        # Ensure numeric fields are ints and compute/validate total_score
        for key in ["communication_clarity", "technical_accuracy", "keyword_usage", "confidence_fluency", "completeness"]:
            try:
                data[key] = int(data.get(key, 0))
            except Exception:
                data[key] = 0

        computed_total = sum(data[k] for k in ["communication_clarity", "technical_accuracy", "keyword_usage", "confidence_fluency", "completeness"])
        try:
            reported_total = int(data.get("total_score", computed_total))
        except Exception:
            reported_total = computed_total

        if reported_total != computed_total:
            print(f"[EVAL] Reported total ({reported_total}) != computed total ({computed_total}). Overriding with computed.")
            data["total_score"] = computed_total
        else:
            data["total_score"] = reported_total

        data.setdefault("correctness", "Partially Correct")
        data.setdefault("feedback", "")
        data.setdefault("improvement", "")
        data.setdefault("difficulty", "Medium")
        data.setdefault("topic_tag", "")

        return data

    except Exception as e:
        print("[EVAL ERROR]", e)
        traceback.print_exc()
        return {
            "communication_clarity": 0,
            "technical_accuracy": 0,
            "keyword_usage": 0,
            "confidence_fluency": 0,
            "completeness": 0,
            "total_score": 0,
            "correctness": "Error",
            "feedback": "Could not parse LLM output.",
            "improvement": "Please retry the evaluation.",
            "difficulty": "Unknown",
            "topic_tag": "Unknown"
        }

# ----------------- Local fallback evaluator ----------------- #
def simple_local_evaluator(question, answer):
    """
    Lightweight fallback: rough heuristics to produce 0-10 integer rubric scores.
    Uses keyword matching and answer length. NOT a replacement for LLM but useful offline.
    """
    def score_by_length(ans):
        n = len(ans.strip().split())
        if n >= 40:
            return 10
        if n >= 30:
            return 8
        if n >= 20:
            return 6
        if n >= 10:
            return 4
        if n > 0:
            return 2
        return 0

    q_words = [w.lower().strip(".,?()") for w in question.split() if len(w) > 3]
    a_lower = answer.lower()
    matched = sum(1 for w in q_words if w in a_lower)
    keyword_score = min(10, int((matched / max(1, len(q_words))) * 10))

    clarity = score_by_length(answer)
    accuracy = keyword_score
    keyword_usage = keyword_score
    fluency = min(10, int(clarity * 0.9))
    completeness = clarity

    total = clarity + accuracy + keyword_usage + fluency + completeness
    return {
        "communication_clarity": clarity,
        "technical_accuracy": accuracy,
        "keyword_usage": keyword_usage,
        "confidence_fluency": fluency,
        "completeness": completeness,
        "total_score": total,
        "correctness": "Partially Correct" if total > 15 else "Incorrect",
        "feedback": "Auto-eval: expand with examples and key terms.",
        "improvement": "Add examples, key terms, and structure the answer.",
        "difficulty": "Medium",
        "topic_tag": ""
    }

# ----------------- Orchestration: run assessment ----------------- #
def run_assessment(topics=TOPICS, q_per_topic=QUESTIONS_PER_TOPIC):
    print("Starting multi-topic assessment. Topics:", topics)
    print("Press Ctrl+C at any time to abort the session.")
    time.sleep(0.7)

    for topic in topics:
        print("\n" + "="*40)
        print(f"TOPIC: {topic}")
        print("="*40 + "\n")

        # generate questions (and ensure they are spoken-answer friendly)
        questions = generate_spoken_questions(topic, n=q_per_topic)
        filtered = []
        for q in questions:
            ok = relevancy_check_question(q, topic)
            if not ok:
                # attempt a safe transformation: ask Gemini to rephrase as spoken-answer question
                try:
                    resp = gemini_model.generate_content(
                        f"Rephrase the following as a spoken-answer friendly question (one sentence). Do NOT ask to write code. Question: {q}"
                    )
                    newq = resp.text.strip()
                    if relevancy_check_question(newq, topic):
                        filtered.append(newq)
                        continue
                except Exception:
                    pass
                # fallback: simple template
                filtered.append(f"Explain this concept in {topic}: {q}")
            else:
                filtered.append(q)

        # remove duplicates while preserving order
        seen = set()
        unique_questions = []
        for q in filtered:
            if q not in seen:
                unique_questions.append(q)
                seen.add(q)
            if len(unique_questions) >= q_per_topic:
                break
        # ensure length
        while len(unique_questions) < q_per_topic:
            unique_questions.append(f"{topic} question — explain a key concept or give an example.")

        # iterate through questions
        for idx, question in enumerate(unique_questions, start=1):
            print(f"\nQuestion {idx}/{q_per_topic} ({topic})")
            print(question)
            # speak
            try:
                speak_text(question, filename=f"q_{topic}_{idx}.wav")
            except Exception as e:
                print("[TTS ERROR]", e)
                traceback.print_exc()
            # record
            audio_file = record_answer(filename=f"answer_{topic}_{idx}.wav")
            # transcribe
            answer_text = speech_to_text(audio_file)
            # evaluate
            eval_result = evaluate_answer_with_gemini(question, answer_text)
            # debug print so you can see exactly what the LLM returned
            print("[EVAL RESULT RAW]", eval_result)

            # If evaluation failed or returned error, use local fallback
            if eval_result.get("correctness") in (None, "Error"):
                print("[EVAL] Using local fallback evaluator.")
                eval_result = simple_local_evaluator(question, answer_text)
                print("[LOCAL EVAL]", eval_result)

            # build row and save (columns in the requested order)
            row = [
                time.strftime("%Y-%m-%d %H:%M:%S"),            # Timestamp
                topic,                                         # Topic
                idx,                                           # Question No
                question,                                      # Question
                answer_text,                                   # Student Answer
                eval_result.get("communication_clarity"),      # Communication Clarity
                eval_result.get("technical_accuracy"),         # Technical Accuracy
                eval_result.get("keyword_usage"),              # Keyword Usage
                eval_result.get("confidence_fluency"),         # Confidence Fluency
                eval_result.get("completeness"),               # Completeness
                eval_result.get("total_score"),                # Total Score (sum 0-50)
                eval_result.get("correctness"),                # Correctness
                eval_result.get("feedback"),                   # Feedback
                eval_result.get("improvement"),                # Improvement
                eval_result.get("difficulty"),                 # Difficulty
                eval_result.get("topic_tag")                   # Topic Tag
            ]
            save_row_to_sheet(row)
            print("\n-- Next question will start shortly --\n")
            time.sleep(0.6)

    print("\nAssessment complete. All done.")

# ----------------- entry point ----------------- #
if __name__ == "__main__":
    try:
        run_assessment()
    except KeyboardInterrupt:
        print("\nSession aborted by user.")
    except Exception as e:
        print("[FATAL ERROR]", e)
        traceback.print_exc()
