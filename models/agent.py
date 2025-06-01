# Written by Juan Pablo Gutiérrez
# 23 01 2025

from dataclasses import dataclass
from cerebraai.models.llm import LLMResponse

class Agent:
    """
    A singleton class representing an AI agent with configurable behavior for handling different types of inputs.
    
    This class implements the Singleton pattern to ensure only one instance exists throughout the application.
    It manages the agent's configuration for file acceptance, text processing, and RAG (Retrieval-Augmented Generation) settings.
    """
    _instance = None 

    accepted_files: list[str]   # List of file extensions/types this agent can process
    accept_text: bool           # Whether the agent can process plain text input
    context_weight: float       # Weight given to context in processing (0.0 to 1.0)
    analysis_weights: dict      # Weight given to text in processing (0.0 to 1.0)
    sentiment_weights: dict     # Weight given to text in processing (0.0 to 1.0)
    emotion_weights: dict       # Weight given to text in processing (0.0 to 1.0)
    rag: bool                   # Whether to use Retrieval-Augmented Generation
    name: str                   # Name identifier for the agen
    
    def __new__(cls, *args, **kwargs):
        """
        Implements the Singleton pattern by ensuring only one instance of Agent exists.
        
        Returns:
            Agent: The single instance of the Agent class
        """
        if not cls._instance:
            cls._instance = super(Agent, cls).__new__(cls)
        return cls._instance

    def __init__(self, name: str, accepted_files: list[str] = [], accept_text: bool = True, rag: bool = True, context_weight: float = 0.5, analysis_weights: dict = {}, sentiment_weights: dict = {}, emotion_weights: dict = {}):
        """
        Initialize the Agent with configuration parameters.
        
        Args:
            name (str): Identifier name for the agent
            accepted_files (list[str]): List of file types/extensions this agent can process
            accept_text (bool): Whether the agent can process plain text input
            rag (bool): Whether to use Retrieval-Augmented Generation
            context_weight (float): Weight given to context in processing (0.0 to 1.0)
            analysis_weights (dict): Weight given to text in processing (0.0 to 1.0)
            sentiment_weights (dict): Weight given to text in processing (0.0 to 1.0)
            emotion_weights (dict): Weight given to text in processing (0.0 to 1.0)
        """
        self.name = name
        self.accepted_files = accepted_files
        self.accept_text = accept_text
        self.rag = rag
        self.context_weight = context_weight
        self.analysis_weights = analysis_weights
        self.sentiment_weights = sentiment_weights
        self.emotion_weights = emotion_weights

    def update_config(self, config: dict):
        try:
            self.name = config["agent_name"]
            self.accepted_files = config["accepted_files"]
            self.accept_text = config["accept_text"]
            self.rag = config["rag"]
            self.context_weight = config["context_weight"]
            self.analysis_weights = config["analysis_weights"]
            self.sentiment_weights = config["sentiment_weights"]
            self.emotion_weights = config["emotion_weights"]
        except KeyError:
            raise KeyError("Invalid config")
    

    def get_accepted_files(self):
        """
        Returns the list of file types this agent can process.
        
        Returns:
            list[str]: List of accepted file types/extensions
        """
        return self.accepted_files

    def get_rag(self):
        """
        Returns whether RAG is enabled for this agent.
        
        Returns:
            bool: True if RAG is enabled, False otherwise
        """
        return self.rag

    def get_accept_text(self):
        """
        Returns whether this agent can process plain text input.
        
        Returns:
            bool: True if text processing is enabled, False otherwise
        """
        return self.accept_text

    def get_context_weight(self):
        """
        Returns the weight given to context in processing.
        
        Returns:
            float: Context weight value between 0.0 and 1.0
        """
        return self.context_weight

    def get_name(self):
        """
        Returns the name identifier of this agent.
        
        Returns:
            str: The agent's name
        """
        return self.name
    
@dataclass
class AgentResponse:
    response: LLMResponse
    response_time: float
    context_time: float = -1.0
    chat_history_time: float = -1.0
    insert_history_time: float = -1.0
    transcription_time: float = 0.0

    def to_dict(self):
        return {
            "response": self.response,
            "transcription_time": self.transcription_time, 
            "response_time": self.response_time,
            "context_time": self.context_time,
            "chat_history_time": self.chat_history_time,
            "insert_history_time": self.insert_history_time
        }