from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
import json
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

# Set up environment variables and models
hf_token = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

if not hf_token or not groq_api_key:
    raise ValueError("Missing required environment variables: HF_TOKEN and/or GROQ_API_KEY")

os.environ["HF_TOKEN"] = hf_token

# Initialize the embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[tuple]] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

class Blog(BaseModel):
    id: Optional[int] = None
    author: str
    designation: str
    date: str
    title: str
    summary: str
    content: str
    rating: Optional[str] = None

class BlogResponse(BaseModel):
    id: int
    author: str
    designation: str
    date: str
    title: str
    summary: str
    content: str
    rating: str

# Initialize vector stores and retriever
try:
    vector_store_paths = {
        "insurance": "vector_store_insurance",
        "health_insurance": "vector_store_health_insurance",
        "user_profile": "vector_store_user_profile"
    }
    
    vector_stores = {}
    missing_stores = []
    
    print("\nChecking vector stores...")
    for name, path in vector_store_paths.items():
        try:
            store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            if store.index.ntotal > 0:
                vector_stores[name] = store
                print(f"✅ Successfully loaded {name} vector store with {store.index.ntotal} documents")
            else:
                missing_stores.append(name)
                print(f"❌ {name} vector store is empty")
        except Exception as e:
            missing_stores.append(name)
            print(f"❌ Error loading {name} vector store: {str(e)}")
    
    if missing_stores:
        print("\n⚠️ Missing or empty vector stores:")
        for store in missing_stores:
            print(f"- {store}")
        print("\nPlease ensure all vector stores are properly created and contain data.")
    
    if not vector_stores:
        raise RuntimeError("No vector stores were successfully loaded. Please check your vector store files.")
    
    class WeightedRetriever(BaseRetriever):
        stores: dict
        weights: dict
        
        def __init__(self, stores: dict, weights: dict):
            super().__init__(stores=stores, weights=weights)
        
        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun | None = None,
        ) -> List[Document]:
            all_docs = []
            
            profile_keywords = ["my profile", "my information", "my details", "about me", 
                              "my health", "my insurance", "my policies", "my coverage"]
            
            is_profile_query = any(keyword in query.lower() for keyword in profile_keywords)
            
            for store_name, store in self.stores.items():
                weight = self.weights.get(store_name, 0.1)
                
                if is_profile_query and store_name == "user_profile":
                    weight = 0.8
                elif not is_profile_query and store_name == "user_profile":
                    weight = 0.1
                
                try:
                    docs = store.similarity_search(query, k=3)
                    for doc in docs:
                        doc.metadata['weight'] = weight
                        doc.page_content = f"[Source: {store_name}]\n{doc.page_content}"
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Warning: Error retrieving documents from {store_name}: {str(e)}")
                    continue
            
            all_docs.sort(key=lambda x: x.metadata.get('weight', 0), reverse=True)
            return all_docs[:5]
    
    retriever = WeightedRetriever(
        stores=vector_stores,
        weights={
            "insurance": 0.3,
            "health_insurance": 0.3,
            "user_profile": 0.4
        }
    )

except Exception as e:
    raise RuntimeError(f"Error setting up vector stores: {e}")

# Create prompt template
system_prompt = """You are an AI Insurance Advisory Assistant specialized in analyzing comprehensive insurance information across multiple domains including general insurance, health insurance, and user profiles. Your role is to provide detailed, accurate information while maintaining context coherence and domain-specific expertise.

When handling user profile queries:
1. Always reference the specific user's information when available
2. Use personal details to provide tailored recommendations
3. Consider the user's financial profile including:
   - Bank account balances
   - Credit card details
   - Mutual fund investments
   - Loan obligations
   - Insurance policies
4. Highlight relevant insurance gaps and needs
5. Provide personalized suggestions based on the user's profile

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Function to update vector store with new blog
def update_vector_store_with_blog(blog: Blog, embeddings):
    try:
        doc = Document(
            page_content=f"Title: {blog.title}\nSummary: {blog.summary}\nContent: {blog.content}\nAuthor: {blog.author}\nDesignation: {blog.designation}\nDate: {blog.date}",
            metadata={
                "source": "blog",
                "id": str(blog.id),
                "title": blog.title,
                "author": blog.author,
                "designation": blog.designation,
                "date": blog.date,
                "rating": blog.rating,
                "created_at": datetime.now().isoformat()
            }
        )
        
        try:
            vector_store = FAISS.load_local("vector_store_insurance", embeddings, allow_dangerous_deserialization=True)
        except:
            vector_store = FAISS.from_documents([doc], embeddings)
        
        vector_store.add_documents([doc])
        vector_store.save_local("vector_store_insurance")
        
        return True
    except Exception as e:
        print(f"Error updating vector store: {e}")
        return False

# Function to reload AI components
def reload_ai_components():
    global rag_chain, retriever, vector_stores
    
    try:
        # Reload vector stores
        vector_stores = {}
        for name, path in vector_store_paths.items():
            try:
                store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
                if store.index.ntotal > 0:
                    vector_stores[name] = store
                    print(f"✅ Successfully reloaded {name} vector store with {store.index.ntotal} documents")
            except Exception as e:
                print(f"❌ Error reloading {name} vector store: {str(e)}")
        
        # Recreate retriever
        retriever = WeightedRetriever(
            stores=vector_stores,
            weights={
                "insurance": 0.3,
                "health_insurance": 0.3,
                "user_profile": 0.4
            }
        )
        
        # Recreate RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        print("✅ AI components successfully reloaded")
        return True
    except Exception as e:
        print(f"❌ Error reloading AI components: {str(e)}")
        return False

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Insurance Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = rag_chain.invoke({
            "input": request.message,
            "chat_history": request.chat_history or []
        })
        
        sources = []
        for doc in response.get("context", []):
            if "source" in doc.metadata:
                sources.append(doc.metadata["source"])
        
        return ChatResponse(
            response=response["answer"],
            sources=list(set(sources))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/blogs", response_model=BlogResponse)
async def create_blog(blog: Blog):
    try:
        # Load existing insurance data
        try:
            with open("dataset/insurance.json", "r") as f:
                insurance_data = json.load(f)
        except:
            insurance_data = []
        
        # Generate new ID if not provided
        if blog.id is None:
            blog.id = len(insurance_data) + 1
        
        # Create blog response
        blog_response = BlogResponse(
            id=blog.id,
            author=blog.author,
            designation=blog.designation,
            date=blog.date,
            title=blog.title,
            summary=blog.summary,
            content=blog.content,
            rating=blog.rating or "0/5"
        )
        
        # Add new blog to insurance data
        insurance_data.append(blog_response.dict())
        
        # Save updated insurance data
        with open("dataset/insurance.json", "w") as f:
            json.dump(insurance_data, f, indent=2)
        
        # Update vector store
        if not update_vector_store_with_blog(blog, embeddings):
            raise HTTPException(status_code=500, detail="Failed to update vector store")
        
        # Reload AI components
        if not reload_ai_components():
            raise HTTPException(status_code=500, detail="Failed to reload AI components")
        
        return blog_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/blogs", response_model=List[BlogResponse])
async def get_blogs():
    try:
        with open("dataset/insurance.json", "r") as f:
            insurance_data = json.load(f)
        return insurance_data
    except:
        return []

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 