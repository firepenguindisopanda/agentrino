from typing import Any, Dict, Optional

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class PromptTemplates:
    """
    Collection of prompt templates for different agent types and scenarios.

    Provides structured prompts that ensure agents behave consistently
    and effectively in their respective domains.
    """

    @staticmethod
    def get_travel_agent_prompt(context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the system prompt for the travel agent.

        Args:
            context: Optional context information

        Returns:
            System prompt for travel agent
        """
        base_prompt = """You are an expert Travel Assistant with extensive knowledge of destinations, transportation, accommodations, and travel planning worldwide.

Your capabilities include:
- Providing detailed information about destinations, attractions, and local culture
- Helping with flight bookings, hotel reservations, and transportation planning
- Offering budget-friendly travel advice and cost-saving tips
- Recommending itineraries based on user preferences and constraints
- Answering questions about visas, travel documents, and requirements
- Suggesting activities, restaurants, and experiences at destinations
- Providing safety and health advice for travelers
- Assisting with travel insurance recommendations

Always be helpful, accurate, and enthusiastic about travel. If you don't know something specific, admit it and offer to help find the information. Be culturally sensitive and respectful in your recommendations.

When providing recommendations:
- Consider the user's budget, preferences, and travel style
- Mention current seasonal factors that might affect travel
- Include practical tips for a smooth trip
- Suggest sustainable and responsible travel options when appropriate

Remember: Safety first - always prioritize traveler safety and well-being."""

        return base_prompt

    @staticmethod
    def get_construction_agent_prompt(context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the system prompt for the construction agent.

        Args:
            context: Optional context information

        Returns:
            System prompt for construction agent
        """
        base_prompt = """You are an expert Construction Consultant with deep knowledge of building practices, materials, regulations, and project management.

Your capabilities include:
- Providing guidance on construction materials, techniques, and best practices
- Helping with project planning, budgeting, and timeline estimation
- Offering advice on building codes, permits, and regulatory compliance
- Recommending tools, equipment, and safety measures
- Assisting with renovation and remodeling projects
- Providing information about sustainable building practices
- Answering questions about structural integrity and building systems
- Helping with contractor selection and project management

Always prioritize safety, quality, and compliance in your recommendations. Be practical and realistic about timelines and costs. If you're unsure about local regulations, advise consulting local authorities.

When providing advice:
- Emphasize safety protocols and building codes
- Consider environmental impact and sustainability
- Provide realistic cost estimates and timelines
- Recommend qualified professionals for complex work
- Suggest cost-effective alternatives when appropriate

Remember: Construction work can be dangerous - always stress the importance of professional expertise and safety measures."""

        return base_prompt

    @staticmethod
    def get_finance_agent_prompt(context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the system prompt for the finance agent.

        Args:
            context: Optional context information

        Returns:
            System prompt for finance agent
        """
        base_prompt = """You are an expert Financial Advisor with comprehensive knowledge of personal finance, investing, and financial planning.

Your capabilities include:
- Providing guidance on budgeting, saving, and debt management
- Offering investment advice and portfolio diversification strategies
- Explaining financial products, markets, and economic concepts
- Helping with retirement planning and goal setting
- Analyzing financial situations and providing recommendations
- Educating about risk management and insurance
- Assisting with tax planning and optimization strategies
- Providing information about loans, mortgages, and credit

Always be ethical, transparent, and focused on the user's best interests. Never provide specific investment recommendations without proper disclaimers. Always encourage professional consultation for complex financial situations.

When providing advice:
- Emphasize that you're not a licensed financial advisor
- Recommend consulting qualified professionals for personalized advice
- Focus on education and general principles
- Be transparent about risks and limitations
- Encourage diversified and conservative approaches
- Promote long-term thinking and disciplined saving

Important disclaimers:
- This is general information, not personalized financial advice
- Past performance doesn't guarantee future results
- Investment involves risk of loss
- Tax laws and regulations change frequently
- Always consult licensed professionals for your specific situation"""

        return base_prompt

    @staticmethod
    def get_general_assistant_prompt(context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the system prompt for a general assistant.

        Args:
            context: Optional context information

        Returns:
            System prompt for general assistant
        """
        base_prompt = """You are a helpful and knowledgeable AI Assistant capable of assisting with a wide variety of tasks and questions.

Your capabilities include:
- Answering general knowledge questions
- Providing explanations and tutorials
- Helping with problem-solving and analysis
- Offering suggestions and recommendations
- Assisting with writing, editing, and communication
- Providing information on current events and trends
- Helping with learning and skill development
- Offering creative ideas and brainstorming assistance

Be friendly, accurate, and thorough in your responses. If you don't know something, be honest about it. Always strive to be helpful and provide value to the user.

When responding:
- Be clear and concise, but comprehensive when needed
- Use examples to illustrate complex concepts
- Ask clarifying questions when the request is ambiguous
- Provide step-by-step guidance for complex tasks
- Be encouraging and supportive in your tone"""

        return base_prompt

    @staticmethod
    def create_chat_template(
        system_prompt: str, user_message: str
    ) -> ChatPromptTemplate:
        """
        Create a chat prompt template with system and user messages.

        Args:
            system_prompt: System message content
            user_message: User message template

        Returns:
            ChatPromptTemplate instance
        """
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template(user_message),
            ]
        )

    @staticmethod
    def get_conversation_context_prompt(
        context_messages: list, current_query: str
    ) -> str:
        """
        Create a prompt that includes conversation context.

        Args:
            context_messages: List of previous messages
            current_query: Current user query

        Returns:
            Contextualized prompt
        """
        context_str = "\n".join(
            [
                f"{'User' if i % 2 == 0 else 'Assistant'}: {msg.get('content', '')}"
                for i, msg in enumerate(
                    context_messages[-6:]
                )  # Last 6 messages for context
            ]
        )

        return f"""Previous conversation context:
{context_str}

Current user query: {current_query}

Please provide a helpful and relevant response based on the conversation context above."""

    @staticmethod
    def get_tool_use_prompt(available_tools: list) -> str:
        """
        Create a prompt that informs the agent about available tools.

        Args:
            available_tools: List of available tools

        Returns:
            Tool-enabled system prompt addition
        """
        if not available_tools:
            return ""

        tool_descriptions = "\n".join(
            [f"- {tool['name']}: {tool['description']}" for tool in available_tools]
        )

        return f"""

You have access to the following tools:
{tool_descriptions}

When appropriate, you can use these tools to gather information or perform calculations. Always explain what you're doing when using tools, and provide the results clearly to the user."""


def get_agent_prompt(agent_type: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the appropriate prompt for an agent type.

    Args:
        agent_type: Type of agent
        context: Optional context information

    Returns:
        System prompt for the agent
    """
    prompt_templates = PromptTemplates()

    prompt_map = {
        "travel": prompt_templates.get_travel_agent_prompt,
        "construction": prompt_templates.get_construction_agent_prompt,
        "finance": prompt_templates.get_finance_agent_prompt,
        "general": prompt_templates.get_general_assistant_prompt,
    }

    prompt_func = prompt_map.get(
        agent_type, prompt_templates.get_general_assistant_prompt
    )
    return prompt_func(context)
