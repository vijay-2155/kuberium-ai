# Insurance Chatbot with FastAPI

A powerful AI-powered insurance advisory system that combines FastAPI for REST endpoints with advanced language models for intelligent responses about insurance topics.

## Features

- **AI-Powered Chatbot**: Uses Groq's LLM for intelligent insurance-related responses
- **REST API**: FastAPI endpoints for chat, blog management, and system health
- **Vector Store Integration**: FAISS-based vector stores for efficient information retrieval
- **Blog Management**: Create and manage insurance-related blog posts
- **Real-time Updates**: Automatic AI model reloading when new content is added
- **Multi-source Knowledge**: Integrates information from multiple insurance domains

## Prerequisites

- Python 3.8+
- Ollama running locally (for embeddings)
- Groq API key
- Hugging Face token

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd aibot
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

## Project Structure

```
aibot/
├── app.py                 # FastAPI application and endpoints
├── main.py               # Core chatbot functionality
├── dataset/
│   ├── insurance.json    # Insurance articles and blogs
│   └── userprofile.json  # User profile data
├── vector_store_insurance/      # FAISS vector store for insurance
├── vector_store_health_insurance/ # FAISS vector store for health insurance
└── vector_store_user_profile/    # FAISS vector store for user profiles
```

## API Endpoints

### Chat
- `POST /chat`: Send messages to the chatbot
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is health insurance?", "chat_history": []}'
```

### Blog Management
- `POST /blogs`: Create a new blog post
```bash
curl -X POST "http://localhost:8000/blogs" \
     -H "Content-Type: application/json" \
     -d '{
         "author": "John Doe",
         "designation": "Insurance Expert",
         "date": "2025-03-01",
         "title": "Understanding Insurance",
         "summary": "A comprehensive guide",
         "content": "Detailed content here...",
         "rating": "4.5/5"
     }'
```

- `GET /blogs`: Get all blog posts
```bash
curl "http://localhost:8000/blogs"
```

### System Health
- `GET /health`: Check system health
```bash
curl "http://localhost:8000/health"
```

## Running the Application

1. Start the FastAPI server:
```bash
python app.py
```

2. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Blog Post Format

Blog posts should follow this structure:
```json
{
    "id": 1,
    "author": "Author Name",
    "designation": "Expert Title",
    "date": "YYYY-MM-DD",
    "title": "Blog Title",
    "summary": "Brief summary",
    "content": "Detailed content",
    "rating": "X.X/5"
}
```

## Vector Stores

The system uses three vector stores:
1. `vector_store_insurance`: General insurance information
2. `vector_store_health_insurance`: Health insurance specific content
3. `vector_store_user_profile`: User profile and personal information

## AI Model Integration

- Uses Groq's LLM for response generation
- Ollama embeddings for document similarity
- FAISS for efficient vector storage and retrieval
- Custom weighted retriever for domain-specific responses

## Error Handling

The system includes comprehensive error handling for:
- Missing environment variables
- Vector store loading failures
- API request validation
- AI model initialization
- Blog creation and updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
mit license

## Contact

vijay-2155
likithsatya192
