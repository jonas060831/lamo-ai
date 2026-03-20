from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import ollama
import os
import json
import hashlib
from utils.sessions_helper import get_or_create_session, cleanup_old_sessions, build_history_context, save_to_history
from utils.generate_tts import generate_tts
from utils.searxng import searxng_search, should_search
from utils.is_coding_question import is_coding_question
from utils.generate_stt import generate_stt
app = Flask(__name__)
CORS(app)

env_name = os.getenv('APP_ENV', 'development')

load_dotenv(f'.env.{env_name}')


PORT = os.getenv('PORT', 8000)

SESSION_TIMEOUT_HOURS = int(os.getenv('SESSION_TIMEOUT_HOURS', 2))

chat_sessions = {}
session_timestamps = {}

#load embeddings and database once server starts
embeddings = HuggingFaceBgeEmbeddings()
db = Chroma(persist_directory="./db", embedding_function=embeddings)

@app.route('/clear-history', methods=['POST'])
def clear_history():
    #clear chat history for a session
    try:
        data = request.get_json()
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({ 'error': 'No session_id provided' }), 400

        if session_id in chat_sessions:
            del chat_sessions[session_id]
            session_timestamps.pop(session_id, None)
            return jsonify({ 'message': 'History cleared', 'session_id': session_id })
        
        return jsonify({ 'message': 'Session not found', 'session_id': session_id })
    
    except Exception as e:
        return jsonify({ 'error': str(e) }), 500

@app.route('/smart-query', methods=['POST'])
def smart_query():
    # Intelligently decides whether to use context or not, with chat history
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data['question'].strip()
        k = data.get('k', 3)
        model = data.get('model', 'llama3')
        threshold = data.get('threshold', 0.5)

        # auto generate session ID if not provided
        session_id = get_or_create_session(chat_sessions, data.get('session_id'))
        session_timestamps[session_id] = __import__('datetime').datetime.now()

        # cleanup old sessions occasionally SESSION_TIMEOUT_HOURS is set to 2 hours
        cleanup_old_sessions(chat_sessions, session_timestamps)

        # Search for relevant documents
        docs = db.similarity_search_with_score(question, k=k)

        # Check if we have relevant context
        has_relevant_context = len(docs) > 0 and docs[0][1] < threshold

        history_context = build_history_context(chat_sessions, session_id)

        if has_relevant_context:

            context = "\n\n".join([doc.page_content for doc, score in docs])

            prompt = f"""
{history_context}
Based on the following context and our conversation history, answer the question.
If the context doesn't contain relevant information, you can answer from your general knowledge.

Context:
{context}

Question: {question}

Answer:
"""

            result = ollama.generate(model=model, prompt=prompt)
            answer = result['response']

            save_to_history(chat_sessions, session_id, question, answer)

            return jsonify({
                "answer": answer,
                "used_context": True,
                "relevance_score": float(docs[0][1]),
                "session_id": session_id
            })

        else:

            # detect if this is a coding question
            coding_question = is_coding_question(question, model)

            if coding_question:
                search_needed = False
            else:
                search_needed = should_search(question, model)

            if search_needed:

                print("SEARCH DECISION:", search_needed)

                web_results = searxng_search(question)

                if not web_results:

                    prompt = f"""
{history_context}
Question: {question}

Answer:
"""

                else:

                    web_context = "\n\n".join([
                        f"{r.get('title','')}\n{r.get('content','')}\n{r.get('url','')}"
                        for r in web_results
                    ])

                    prompt = f"""
{history_context}

Use the following web search results to answer the question.

Web Results:
{web_context}

Question: {question}

Answer:
"""

                result = ollama.generate(model=model, prompt=prompt)
                answer = result['response']

                save_to_history(chat_sessions, session_id, question, answer)

                return jsonify({
                    "answer": answer,
                    "used_context": False,
                    "used_web_search": True,
                    "session_id": session_id
                })

            else:

                prompt = f"""
{history_context}
Question: {question}

Answer:
"""

                result = ollama.generate(model=model, prompt=prompt)
                answer = result['response']

                save_to_history(chat_sessions, session_id, question, answer)

                return jsonify({
                    "answer": answer,
                    "used_context": False,
                    "used_web_search": False,
                    "session_id": session_id
                })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voice-query', methods=['POST'])
def voice_query():
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data['question']
        k = data.get('k', 3)
        model = data.get('model', 'mistral')
        threshold = data.get('threshold', 0.5)

        session_id = get_or_create_session(chat_sessions, data.get('session_id'))
        session_timestamps[session_id] = __import__('datetime').datetime.now()

        cleanup_old_sessions(chat_sessions, session_timestamps)

        docs = db.similarity_search_with_score(question, k=k)

        has_relevant_context = len(docs) > 0 and docs[0][1] < threshold
        history_context = build_history_context(chat_sessions, session_id)

        if has_relevant_context:

            context = "\n\n".join([doc.page_content for doc, score in docs])

            prompt = f"""
{history_context}
Based on the following context and our conversation history, answer the question.
If the context doesn't contain relevant information, you can answer from your general knowledge.

Context:
{context}

Question: {question}

Answer:
"""

            result = ollama.generate(model=model, prompt=prompt)
            answer = result['response']

            save_to_history(chat_sessions, session_id, question, answer)
            audio_base64 = generate_tts(answer)

            return jsonify({
                "answer": answer,
                "audio": audio_base64,
                "used_context": True,
                "relevance_score": float(docs[0][1]),
                "session_id": session_id
            })

        else:

            # detect if this is a coding question
            coding_question = is_coding_question(question, model)

            if coding_question:
                search_needed = False
            else:
                search_needed = should_search(question, model)

            if search_needed:

                print("SEARCH DECISION:", search_needed)

                web_results = searxng_search(question)

                if not web_results:
                    prompt = f"""
{history_context}
Question: {question}

Answer:
"""
                else:
                    web_context = "\n\n".join([
                        f"{r.get('title','')}\n{r.get('content','')}\n{r.get('url','')}"
                        for r in web_results
                    ])

                    prompt = f"""
{history_context}

Use the following web search results to answer the question.

Web Results:
{web_context}

Question: {question}

Answer:
"""

                result = ollama.generate(model=model, prompt=prompt)
                answer = result['response']
                audio_base64 = generate_tts(answer)

                save_to_history(chat_sessions, session_id, question, answer)

                return jsonify({
                    "answer": answer,
                    "audio": audio_base64,
                    "used_context": False,
                    "used_web_search": True,
                    "session_id": session_id
                })

            else:

                prompt = f"""
{history_context}
Question: {question}

Answer:
"""

                result = ollama.generate(model=model, prompt=prompt)

                answer = result['response']
                audio_base64 = generate_tts(answer)

                save_to_history(chat_sessions, session_id, question, answer)

                return jsonify({
                    "answer": answer,
                    "audio": audio_base64,
                    "used_context": False,
                    "used_web_search": False,
                    "session_id": session_id
                })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔹 Simple in-memory cache: key = hash of text, value = parsed result
ai_cache = {}

@app.route('/detect-ai', methods=['POST'])
def detect_ai():
    MIN_CHARS = 80
    MIN_WORDS = 15

    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text'].strip()
        num_reasons = data.get('num_reasons', 2)  # default to 2 if not provided

        # 🛑 LENGTH GUARD
        word_count = len(text.split())
        char_count = len(text)

        if char_count < MIN_CHARS or word_count < MIN_WORDS:
            return jsonify({
                "error": "Text too short for reliable AI detection",
                "details": {
                    "characters": char_count,
                    "words": word_count,
                    "minimum_characters": MIN_CHARS,
                    "minimum_words": MIN_WORDS
                }
            }), 400

        # 🔑 create a hash of the text to use as cache key
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # 🏷 Check cache first
        if text_hash in ai_cache:
            return jsonify({
                "cached": True,
                **ai_cache[text_hash]
            })

        # 📝 Prompt for the model
        prompt = f"""
You are an expert AI text detector.

Analyze the text and estimate how likely it is AI-generated.

Return ONLY valid JSON with this exact structure:

{{
  "ai_probability": number (0-100),
  "confidence": number (0-100),
  "signals": [
    "reason 1",
    "reason 2"
  ]
}}

Rules:
- Provide EXACTLY {num_reasons} signals
- Each signal must be short and specific
- No extra text outside JSON

Text:
\"\"\"{text}\"\"\"
"""

        # 🔹 Call Ollama
        response = ollama.generate(model="llama3", prompt=prompt)
        answer = response['response']

        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError:
            return jsonify({
                "error": "Failed to parse model response",
                "raw": answer
            }), 500

        # 🔒 enforce exact number of reasons
        signals = parsed.get("signals", [])
        if len(signals) > num_reasons:
            signals = signals[:num_reasons]
        elif len(signals) < num_reasons:
            signals += ["Insufficient signal detected"] * (num_reasons - len(signals))
        parsed["signals"] = signals

        # 🏷 Save result in cache
        ai_cache[text_hash] = parsed

        return jsonify({
            "cached": False,
            **parsed
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#file type .webm
@app.route('/transcribe-audio', methods=['POST'])
def transcribe_audio():
    
    try:
        audio = request.files.get('audio') #from FormData

        if not audio:
            return jsonify({ "error": "No audio uploaded" }), 400
        
        text = generate_stt(audio)
        
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)