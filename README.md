This project implements a **recipe extraction agent** using the **Langchain** framework. It is designed to **extract structured recipe information** such as the title, ingredients, instructions, cooking time, and servings from unstructured text, with a focus on **processing transcripts from cooking YouTube videos**.



## Features

*   **Extracts key recipe details**: Identifies the recipe title, URL (if available), ingredients with quantities, step-by-step instructions, cooking time, and the number of servings 
*   **Processes YouTube transcripts**: Specifically built to analyze and extract information from YouTube video transcripts 
*   **Handles messy or incomplete transcripts**: Attempts to make educated guesses about missing information and provides reasonable estimates for vague measurements 
*   **Outputs in Markdown format**: The extracted recipe is formatted as a clean and readable Markdown string
*   **Utilizes Langchain**: Leverages the power of Langchain for text processing, prompting, and agent creation 
*   **Integrates with OpenAI**: Uses OpenAI models (like `gpt-4o`) for understanding and extracting recipe information
*   **Includes tools for YouTube search**: Can search YouTube for relevant cooking videos

## Installation

To use this recipe extraction agent, you'll need to follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/raym26/recipe-extraction-agent-langchain.git
    cd recipe-extraction-agent-langchain
    ```

2.  **Install the required Python packages:**
    It's recommended to use a virtual environment. If you have a `requirements.txt` file, you can install all dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(Based on our previous conversation about installing `requirements.txt` in Colab)*

3.  **Set up environment variables:**
    This project likely requires an OpenAI API key. You might need to create a `.env` file in the project root and add your API key:
    ```
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY
    ```
    *(Implied by the loading of environment variables in source [4] and the use of `OpenAIEmbeddings` and `ChatOpenAI` in sources [2, 4])*

