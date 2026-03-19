from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub  # or use OpenAI
from .image_processor import ImageProcessor

class MultimodalAgent:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Define tools the agent can use
        self.tools = [
            Tool(
                name="Image Captioning",
                func=self._get_caption,
                description="Use this to get a general description of what's in the image"
            ),
            Tool(
                name="Object Detection",
                func=self._get_objects,
                description="Use this to find specific objects in the image and their locations"
            )
        ]
        
        # Initialize LLM (using HuggingFace for free tier)
        self.llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            model_kwargs={"temperature": 0.5, "max_length": 500}
        )
        
        # Create the agent
        prompt = PromptTemplate.from_template("""
        You are a helpful AI assistant that answers questions about images.
        You have access to tools that can analyze images.
        
        Chat History: {chat_history}
        Question: {input}
        
        Use the tools below to help answer the question:
        {tool_names}
        {tools}
        
        Think step by step:
        1. What information do I need to answer this question?
        2. Which tool should I use first?
        3. What did the tool tell me?
        4. Is there enough information, or do I need another tool?
        
        {agent_scratchpad}
        """)
        
        self.agent = create_react_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _get_caption(self, _) -> str:
        """Tool function to get image caption"""
        return self.image_processor.get_caption(self.current_image)
    
    def _get_objects(self, _) -> str:
        """Tool function to get detected objects"""
        objects = self.image_processor.detect_objects(self.current_image)
        return "\n".join([f"- {obj['object']} (confidence: {obj['confidence']})" 
                         for obj in objects])
    
    def process_query(self, image: Image.Image, query: str) -> str:
        """Main entry point - process a user query about an image"""
        self.current_image = image
        response = self.executor.invoke({"input": query})
        return response["output"]