# Import necessary libraries and modules
from crewai import Crew, Agent, Task  # Import core CrewAI components
from langchain.tools import Tool  # Import Tool from langchain
from typing import List  # Import List for type hinting
import json  # Import json for handling JSON data
from datetime import datetime  # Import datetime for time-related operations
import requests  # Import requests for HTTP requests
from langchain_community.llms.ollama import Ollama as LLM  # Import Ollama for LLM integration
from IPython.display import Markdown


# Initialize the Ollama LLM with specific configuration
llm = LLM(
    model="qwen2.5:latest",  # Specify the model to use
    base_url="http://localhost:11434"  # Set the base URL for Ollama server
)

# Define ContentPlanner Agent class
class ContentPlanner(Agent):
    def __init__(self, topic: str):
        # Initialize ContentPlanner with inherited Agent properties
        super().__init__(
            role="Content Planner",  # Set the agent's role
            goal=f"Plan engaging and factually accurate content on {topic}",  # Define the agent's goal
            backstory=f"You're working on planning a blog article "  # Provide context and background
                     f"about the topic: {topic}."
                     f"You collect information that helps the "
                     f"audience learn something "
                     f"and make informed decisions. "
                     f"Your work is the basis for "
                     f"the Content Writer to write an article on this topic.",
            allow_delegation=False,  # Disable task delegation
            llm=llm,  # Pass the LLM instance
            verbose=True  # Enable verbose output
        )

# Define ContentWriter Agent class
class ContentWriter(Agent):
    def __init__(self, topic: str):
        # Initialize ContentWriter with inherited Agent properties
        super().__init__(
            role="Content Writer",  # Set the agent's role
            goal=f"Write insightful and factually accurate opinion piece about the topic: {topic}",  # Define the goal
            backstory=f"You're working on writing a new opinion piece about the topic: {topic}. "  # Set the context
                    f"You base your writing on the work of the Content Planner, who provides an outline "
                    f"and relevant context about the topic. You follow the main objectives and "
                    f"direction of the outline, as provide by the Content Planner. "
                    f"You also provide objective and impartial insights and back them up with information "
                    f"provide by the Content Planner. You acknowledge in your opinion piece "
                    f"when your statements are opinions as opposed to objective statements.",
            allow_delegation=False,  # Disable task delegation
            llm=llm,  # Pass the LLM instance
            verbose=True  # Enable verbose output
        )

# Define ContentEditor Agent class 
class ContentEditor(Agent):
    def __init__(self):
        super().__init__(
            role="Editor",
            goal="Edit a given blog post to align with "
                 "the writing style of the organization. ",
            backstory="You are an editor who receives a blog post "
                      "from the Content Writer. "
                      "Your goal is to review the blog post "
                      "to ensure that it follows journalistic best practices,"
                      "provides balanced viewpoints "
                      "when providing opinions or assertions, "
                      "and also avoids major controversial topics "
                      "or opinions when possible.",
            allow_delegation=False,
            llm=llm,  # Pass the LLM instance
            verbose=True
        )

# a Task class for ContentPlanner
class PlannerTask(Task):
    def __init__(self, topic: str, agent: Agent):
        super().__init__(
            description=(
                 "1. Prioritize the latest trends, key players, "
                f"and noteworthy news on {topic}.\n"
                "2. Identify the target audience, considering "
                "their interests and pain points.\n"
                "3. Develop a detailed content outline including "
                "an introduction, key points, and a call to action.\n"
                 "4. Include SEO keywords and relevant data or sources."
            ),
             expected_output="A comprehensive content plan document "
                "with an outline, audience analysis, "
                "SEO keywords, and resources.",
             agent=agent
        )

# a Task class for ContentWriter
class WriterTask(Task):
    def __init__(self, topic: str, agent: Agent):
        super().__init__(
            description=(
                "1. Use the content plan to craft a compelling "
                f"blog post on {topic}.\n"
                "2. Incorporate SEO keywords naturally.\n"
                "3. Sections/Subtitles are properly named "
                "in an engaging manner.\n"
                "4. Ensure the post is structured with an "
                "engaging introduction, insightful body, "
                "and a summarizing conclusion.\n"
                "5. Proofread for grammatical errors and "
                "alignment with the brand's voice.\n"
            ),
            expected_output="A well-written blog post "
                "in markdown format, ready for publication, "
                "each section should have 2 or 3 paragraphs.",
            agent=agent
        )

# a Task class for ContentEditor
class EditorTask(Task):
    def __init__(self, topic: str, agent: Agent):
        super().__init__(
            description=(
                "1. Review the blog post content for accuracy, "
                "clarity, and coherence.\n"
                 "2. Check for proper grammar, punctuation, "
                "and formatting.\n"
                "3. Ensure alignment with brand voice and style.\n"
                "4. Verify balanced viewpoint presentation.\n"
                "5. Make necessary edits while preserving the "
                "original message."
            ),
            expected_output="A polished, publication-ready blog post "
                "that maintains quality standards and "
                "aligns with brand guidelines.",
            agent=agent
        )


# Define a function to create a Crew instance
def createCrew(topic: str) -> Crew:
    # Create instances of agents
    content_planner = ContentPlanner(topic)
    content_writer = ContentWriter(topic)
    content_editor = ContentEditor()

    # Create a Crew instance with the agents
    crew = Crew(
        agents=[content_planner, content_writer, content_editor],
        tasks=[
            PlannerTask(topic=topic, agent=content_planner),
            WriterTask(topic=topic, agent=content_writer),
            EditorTask(topic=topic, agent=content_editor)
        ]
    )
    return crew



# Main execution block
if __name__ == "__main__":
    # Create an instance of ContentPlanner with a specific topic
    topic = "Artificial Intelligence in Healthcare"
    crew = createCrew(topic)
    result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})
    # save the result in a markdown file (.md)
    with open(f"{topic.replace(' ', '_')}.md", 'w', encoding='utf-8') as f:
        f.write(result)
    