from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from typing import TypedDict, Annotated, Sequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from utils.config import Config

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class RAGAgent:
    def __init__(self, retriever_tool):
        self.llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL, temperature=0)
        self.tools = [retriever_tool]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.graph = self._build_graph()
    
    def _should_continue(self, state: AgentState) -> str:
        """Check if the last message contains any tool calls."""
        result = state["messages"][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
    
    def _call_llm(self, state: AgentState) -> AgentState:
        """Function to call the LLM with the current state."""
        system_prompt = """
        You are an intelligent AI assistant that answers questions based on the uploaded PDF document.
        Use the retriever tool to search for relevant information in the document. You can make multiple calls if needed.
        Always provide accurate, context-based answers and cite relevant information from the document.
        If the information is not found in the document, clearly state that.
        """
        
        messages = list(state['messages'])
        messages = [SystemMessage(content=system_prompt)] + messages
        message = self.llm_with_tools.invoke(messages)
        return {'messages': [message]}
    
    def _take_action(self, state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response."""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        
        for t in tool_calls:
            if not t['name'] in self.tools_dict:
                result = "Tool not available."
            else:
                result = self.tools_dict[t['name']].invoke(t['args'])
            
            results.append(ToolMessage(
                tool_call_id=t['id'], 
                name=t['name'], 
                content=str(result)
            ))
        
        return {'messages': results}
    
    def _build_graph(self):
        """Build the LangGraph state machine."""
        graph = StateGraph(AgentState)
        graph.add_node("llm", self._call_llm)
        graph.add_node("action", self._take_action)
        
        graph.add_conditional_edges(
            "llm",
            self._should_continue,
            {True: "action", False: END}
        )
        
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        
        return graph.compile()
    
    def query(self, question: str):
        """Query the RAG agent with a question."""
        messages = [HumanMessage(content=question)]
        result = self.graph.invoke({"messages": messages})
        return result['messages'][-1].content