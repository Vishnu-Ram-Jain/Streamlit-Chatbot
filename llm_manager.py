import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

class ChatbotManager:
    def __init__(self):
        # Initialize the Hugging Face Endpoint model
        self.model = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct", 
            task="text-generation",
            temperature=0.7,
            max_new_tokens=512,
            huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )
        # Wrap it inside the ChatHuggingFace interface for system message compatibility
        self.llm = ChatHuggingFace(llm=self.model)
        
        # Define the system prompt and conversation placeholders
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly, witty AI companion. Answer concisely and creatively."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{text}")
        ])
        
        # Build the functional LangChain core pipeline
        self.parser = StrOutputParser()
        self.llm_chain = self.prompt | self.llm | self.parser 
        
        # Dictionary to store session histories globally within the runtime memory
        self.database_sessions = {}
        
        # Wrap everything to auto-manage session compilation
        self.executor = RunnableWithMessageHistory(
            self.llm_chain,          
            self.get_session_history,         
            input_messages_key="text",
            history_messages_key="chat_history"
        )

    def get_session_history(self, session_id: str):
        """Retrieves or creates a chat history track for a specific session ID."""
        if session_id not in self.database_sessions:
            self.database_sessions[session_id] = InMemoryChatMessageHistory()
        return self.database_sessions[session_id]

    def get_response(self, user_input: str, session_id: str) -> str:
        """Invokes the chain while sending traces to LangSmith automatically via Env vars."""
        config = {"configurable": {"session_id": session_id}}
        return self.executor.invoke({"text": user_input}, config=config)
