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

# Load individual vector stores
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
            # Verify the store has documents
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
        print("You can create them by running the vector store creation script.")
    
    if not vector_stores:
        raise RuntimeError("No vector stores were successfully loaded. Please check your vector store files.")
    
    # Create a custom retriever class
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
            
            # Check if query is related to user profile
            profile_keywords = ["my profile", "my information", "my details", "about me", 
                              "my health", "my insurance", "my policies", "my coverage"]
            
            is_profile_query = any(keyword in query.lower() for keyword in profile_keywords)
            
            for store_name, store in self.stores.items():
                weight = self.weights.get(store_name, 0.1)
                
                # Adjust weights based on query type
                if is_profile_query and store_name == "user_profile":
                    weight = 0.8  # Higher weight for user profile queries
                elif not is_profile_query and store_name == "user_profile":
                    weight = 0.1  # Lower weight for non-profile queries
                
                try:
                    docs = store.similarity_search(query, k=3)
                    for doc in docs:
                        doc.metadata['weight'] = weight
                        # Add source information to the content
                        doc.page_content = f"[Source: {store_name}]\n{doc.page_content}"
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Warning: Error retrieving documents from {store_name}: {str(e)}")
                    continue
            
            all_docs.sort(key=lambda x: x.metadata.get('weight', 0), reverse=True)
            return all_docs[:5]
    
    # Create the weighted retriever instance with adjusted weights
    retriever = WeightedRetriever(
        stores=vector_stores,
        weights={
            "insurance": 0.3,
            "health_insurance": 0.3,
            "user_profile": 0.4  # Higher default weight for user profile
        }
    )

except Exception as e:
    raise RuntimeError(f"Error setting up vector stores: {e}")

# Create prompt template with enhanced user profile handling
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

For financial queries:
1. Provide exact account balances when available
2. Include transaction history if relevant
3. Consider overall financial health
4. Relate financial status to insurance needs
5. Maintain confidentiality of sensitive information

DOMAIN EXPERTISE:
1. General Insurance
   - Insurance technology and innovation
   - IRDAI reforms and digital initiatives
   - FDI, market investments
   - Cybersecurity and reinsurance
   - Insurance penetration and market trends

2. Health Insurance
   - Health insurance policies and coverage
   - Medical insurance claims process
   - Health insurance regulations
   - Digital health initiatives
   - Preventive healthcare coverage

3. User Profiles & Personal Information
   - Personal Details
     * Name and contact information
     * Date of birth
     * PAN number
   
   - Financial Accounts
     * Bank account balances
     * Credit card details
     * Mutual fund investments
     * Loan obligations
   
   - Insurance Portfolio
     * Current policies
     * Coverage amounts
     * Premium details
     * Nominee information

RESPONSE STRUCTURE:
1. Domain Identification
   - Identify the primary domain of the query
   - Cross-reference relevant information across domains
   - Maintain domain-specific context

2. Information Analysis
   - Extract key information from relevant posts
   - Identify patterns and trends
   - Connect related concepts across domains
   - Maintain chronological relevance

3. Response Format
   📌 Title/Subject
   📊 Key Statistics (if applicable)
   📑 Main Discussion Points (2-3 key points)
   💡 Practical Implications
   ⚠️ Important Considerations/Warnings
   🔍 Related Topics (if relevant)
   👤 Personal Context (when relevant)

4. Response Rules:
   - Start with "Based on the available information..."
   - Provide exact financial figures when available
   - Include relevant transaction history
   - Maintain professional, clear language
   - Focus on factual information only
   - Acknowledge information gaps explicitly
   - Suggest related topics when appropriate
   - Handle personal information with confidentiality
   - Consider user's specific circumstances in recommendations

5. Special Cases:
   - For financial queries: 
     * Provide exact account balances
     * Include recent transactions if relevant
     * Consider overall financial health
     * Relate to insurance needs
   
   - For insurance queries: 
     * Consider current coverage
     * Evaluate financial capacity
     * Suggest appropriate coverage levels
     * Highlight gaps in coverage
   
   - For investment queries: 
     * Consider current investments
     * Evaluate risk profile
     * Suggest diversification
     * Relate to financial goals

6. When Information is Unavailable:
   - Respond with: "I apologize, but the available information does not contain specific details about [query topic]."
   - Suggest related topics that might be helpful
   - Offer to explore alternative aspects of the query
   - Request additional information if needed for personalized recommendations

7. Personal Information Handling:
   - Maintain strict confidentiality
   - Use personal data only for relevant recommendations
   - Never share sensitive information
   - Consider privacy regulations
   - Provide opt-out options for data usage

Remember:
- Maintain accuracy and context completeness
- Be concise but comprehensive
- Focus on practical, actionable insights
- Avoid speculation or personal opinions
- Stay within insurance industry scope
- Consider the user's perspective and needs
- Handle personal information with utmost care
- Provide personalized recommendations when possible
- Respect user privacy and data protection

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def get_response(question: str, chat_history: list = None) -> str:
    """
    Get a response from the RAG chain for a given question.
    
    Args:
        question (str): The question to ask
        chat_history (list, optional): List of previous messages. Defaults to None.
    
    Returns:
        str: The response from the RAG chain
    """
    try:
        response = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history or []
        })
        return response["answer"]
    except Exception as e:
        return f"Error processing your request: {str(e)}"

def live_chat():
    """
    Start a live chat session with the chatbot.
    Users can type their questions and get responses until they type 'exit' or 'quit'.
    """
    print("\nWelcome to the Insurance Chatbot!")
    print("Type your questions about insurance, and I'll help you find relevant information.")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    chat_history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye! Have a great day!")
                break
                
            if not user_input:
                print("Please enter a question.")
                continue
                
            print("\nBot: ", end="", flush=True)
            response = get_response(user_input, chat_history)
            print(response)
            
            # Update chat history
            chat_history.append(("human", user_input))
            chat_history.append(("ai", response))
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Have a great day!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")

# Example usage
if __name__ == "__main__":
    live_chat()