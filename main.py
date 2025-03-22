import os # For environment variables
from dotenv import load_dotenv # For loading environment variables from a .env file
from typing import List, Optional # For type hinting in the RecipeExtraction model

# Langchain modules
from langchain_community.document_loaders import YoutubeLoader # Loader for YouTube transcripts
from langchain_openai import ChatOpenAI # OpenAI chat model
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder # For creating prompts
from langchain_core.output_parsers import PydanticOutputParser # For parsing language model output into structured formats
from pydantic import BaseModel, Field # For defining the RecipeExtraction data structure
from langchain.memory import ConversationBufferMemory # For storing conversation history
from langchain_community.tools import YouTubeSearchTool # Tool for searching YouTube
from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool # For creating and running agents
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Create a ChatOpenAI model instance
model = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ.get("OPENAI_API_KEY")) # Ensure you have OPENAI_API_KEY in your .env file


# Define the output structure for a recipe using Pydantic
class RecipeExtraction(BaseModel):
    title: str = Field(description="The title or name of the recipe")
    Url: Optional[str] = Field(description="The URL of the recipe", default="")
    ingredients: List[str] = Field(description="List of ingredients with quantities")
    instructions: List[str] = Field(description="Step-by-step cooking instructions")
    cookingTime: str = Field(description="Total time needed to prepare and cook the recipe")
    servings: str = Field(description="Number of servings the recipe makes")


def to_markdown(self) -> str:
    """Convert the recipe to a markdown formatted string with bold headings."""
    markdown = f"**# {self.title}**\n\n"
    if self.Url:
        markdown += f"**URL:** {self.Url}\n\n"
    markdown += f"**Cooking Time:** {self.cookingTime} \n"
    markdown += f"**Servings:** {self.servings}\n\n"
    markdown += "**## Ingredients**\n\n"
    for ingredient in self.ingredients:
        markdown += f"- {ingredient}\n"
    markdown += "\n**## Instructions**\n\n"
    for i, step in enumerate(self.instructions, 1):
        markdown += f"{i}. {step}\n"
    return markdown


RecipeExtraction.to_markdown = to_markdown  # Add the to_markdown method to the RecipeExtraction class

# Create a parser to structure the output of the language model
parser = PydanticOutputParser(pydantic_object=RecipeExtraction)

# Create a memory object to store the agent's conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Create the recipe extraction function
def extract_recipe(transcript):
    """Extracts recipe information from a YouTube transcript."""
    # Simple prompt for the extraction function
    extraction_prompt = PromptTemplate(
        template="""You are a professional chef who specializes in extracting clear recipe information from unstructured text.

Below is a transcript extracted from a cooking YouTube video. Please analyze it carefully and extract the following information:

- The recipe title and url

- A complete list of ingredients with measurements

- Clear step-by-step cooking instructions

- Cooking time information

- Number of servings

Even if the transcript is messy or incomplete, use your expertise to make educated guesses about missing information. If measurements are vague, provide reasonable estimates.

YOUTUBE TRANSCRIPT:

{transcript}

{format_instructions}

""",
        input_variables=["transcript"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    prompt_value = extraction_prompt.format(transcript=transcript)
    output = model.invoke(prompt_value)
    try:
        recipe = parser.parse(output.content)
        return recipe.to_markdown()
    except Exception as e:
        return f"Error parsing recipe: {e}\nOutput content: {output.content}"


# Create the recipe extraction tool
recipe_tool = Tool(
    name="RecipeExtractor",
    func=extract_recipe,
    description="Extracts structured recipe information (ingredients and instructions) from a YouTube video transcript and returns it in markdown format. Input should be the raw transcript text."
)

# Create the YouTube search tool
youtube_tool = YouTubeSearchTool()

# Define the list of tools available to the agent
tools = [
    youtube_tool,
    recipe_tool
]

# Create the prompt for the agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can find and extract cooking recipes from YouTube videos."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the agent
agent = create_tool_calling_agent(
    llm=model,
    tools=tools,
    prompt=agent_prompt

)

# Create the agent executor to run the agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
)

# Define your recipe query
recipe_query = "Find me a  recipe for carbonara and extract the detailed instructions"

# Invoke the agent with the query
response = agent_executor.invoke(
    {
        "input": recipe_query
    },
    config={
        "tool_configs": [
            {
                "name": youtube_tool.name,
                "num_results": 1  # Specify the desired number of results here
            }
        ]
    }
)

# Print the agent's response
print(response["output"])