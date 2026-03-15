from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, after_this_request, send_file
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import ollama
import os
from utils.sessions_helper import get_or_create_session, cleanup_old_sessions, build_history_context, save_to_history
import uuid
from TTS.api import TTS

app = Flask(__name__)
CORS(app)
PORT = os.getenv('PORT', 3000)

SESSION_TIMEOUT_HOURS = int(os.getenv('SESSION_TIMEOUT_HOURS', 2))

chat_sessions = {}
session_timestamps = {}

#load embeddings and database once server starts
embeddings = HuggingFaceBgeEmbeddings()
db = Chroma(persist_directory="./db", embedding_function=embeddings)


# this model allow you to use a different voice
tts = TTS(model_name="tts_models/en/vctk/vits")

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
    #Intelligently decides whether to use context or not, with chat history
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question']
        k = data.get('k', 3)
        model = data.get('model', 'mistral')
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
    
@app.route('/speak', methods=["POST"])
def speak():

    text = request.json["text"]
    filename = f"{uuid.uuid4()}.wav"

    tts.tts_to_file(text=text,speaker="p251", file_path=filename)

     # delete the file after
    @after_this_request
    def remove_file(response):
        try:
            os.remove(filename)
        except Exception as e:
            print("Error deleting file: ", e)
        return response

    return send_file(filename, mimetype="audio/wav")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)