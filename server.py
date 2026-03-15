from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import ollama
import os
from utils.sessions_helper import get_or_create_session, cleanup_old_sessions, build_history_context, save_to_history
from utils.generate_tts import generate_tts
from utils.searxng import searxng_search, should_search
from utils.is_coding_question import is_coding_question
app = Flask(__name__)
CORS(app)
PORT = os.getenv('PORT', 3000)

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
    #Intelligently decides whether to use context or not, with chat history
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question']
        k = data.get('k', 3)
        model = data.get('model', 'llama3')
        threshold = data.get('threshold', 0.5)
        
        #auto generate session ID if not provided
        session_id = get_or_create_session(chat_sessions, data.get('session_id'))
        session_timestamps[session_id] = __import__('datetime').datetime.now()

        #cleanup old sessions occasionally SESSION_TIMEOUT_HOURS is set to 2 hours
        cleanup_old_sessions(chat_sessions, session_timestamps)

        # Search for relevant documents
        docs = db.similarity_search_with_score(question, k=k)
        
        # Check if we have relevant context
        has_relevant_context = len(docs) > 0 and docs[0][1] < threshold
        history_context = build_history_context(chat_sessions, session_id)
        
        if has_relevant_context:
            # Use context from training data
            context = "\n\n".join([doc.page_content for doc, score in docs])
            prompt = f"{history_context}Based on the following context and our conversation history, answer the question. If the context doesn't contain relevant information, you can answer from your general knowledge.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            result = ollama.generate(model=model, prompt=prompt)
            
            save_to_history(chat_sessions, session_id, question, result['response'])
            
            return jsonify({
                "answer": result['response'],
                "used_context": True,
                "relevance_score": float(docs[0][1]),
                "session_id": session_id
            })
        else:
            # No relevant context, just answer generally with history
            prompt = f"{history_context}Question: {question}\n\nAnswer:"
            result = ollama.generate(model=model, prompt=prompt)
            
            save_to_history(chat_sessions, session_id, question, result['response'])
            
            return jsonify({
                "answer": result['response'],
                "used_context": False,
                "message": "No relevant training data found, answered from general knowledge",
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
                audio_base64 = generate_tts(answer)

                save_to_history(chat_sessions, session_id, question, answer)

                return jsonify({
                    "answer": answer,
                    "audio": audio_base64,
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)