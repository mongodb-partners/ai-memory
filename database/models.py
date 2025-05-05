import datetime
from fastapi import HTTPException
from services.bedrock_service import generate_embedding

class Message:
    def __init__(self, message_data):
        self.user_id = message_data.user_id.strip()
        self.conversation_id = message_data.conversation_id.strip()
        self.type = message_data.type
        self.text = message_data.text.strip()
        self.timestamp = self.parse_timestamp(message_data.timestamp)
        self.embeddings = generate_embedding(self.text)
        
    def parse_timestamp(self, timestamp):
        if timestamp:
            try:
                return datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid timestamp format")
        return datetime.datetime.now(datetime.timezone.utc)
        
    def to_dict(self):
        return {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "type": self.type,
            "text": self.text,
            "timestamp": self.timestamp,
            "embeddings": self.embeddings,
        }