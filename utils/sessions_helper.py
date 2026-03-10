from datetime import datetime, timedelta
import os
import uuid

SESSION_TIMEOUT_HOURS = int(os.getenv('SESSION_TIMEOUT_HOURS', 2))

def cleanup_old_sessions(chat_sessions, session_timestamps):
    now = datetime.now()
    expired = [
        sid for sid, ts in session_timestamps.items()
        if now - ts > timedelta(hours=SESSION_TIMEOUT_HOURS)
    ]
    for sid in expired:
        del chat_sessions[sid]
        del session_timestamps[sid]
    if expired:
        print(f"🧹 Cleaned up {len(expired)} expired sessions")


def save_to_history(chat_sessions, session_id, question, answer):
    chat_sessions[session_id].append({'role': 'User', 'content': question})
    chat_sessions[session_id].append({'role': 'Assistant', 'content': answer})
    # Keep only last 50 messages (25 exchanges)
    if len(chat_sessions[session_id]) > 50:
        chat_sessions[session_id] = chat_sessions[session_id][-50:]


def build_history_context(chat_sessions, session_id):
    if not chat_sessions.get(session_id):
        return ""
    history = "Previous conversation:\n"
    for msg in chat_sessions[session_id][-6:]:
        history += f"{msg['role']}: {msg['content']}\n"
    return history + "\n"


def get_or_create_session(chat_sessions, session_id=None):
    if not session_id or session_id not in chat_sessions:
        session_id = session_id or str(uuid.uuid4())
        chat_sessions[session_id] = []
    return session_id