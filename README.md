# RegNavigator Lite

This Streamlit application demonstrates how to use a GenAI-powered pipeline to analyze regulatory dockets from regulations.gov. It follows the "reframe work as code" methodology to quickly turn a complex policy analysis task into an automated, interactive dashboard.

## Setup for macOS

### Prerequisites
*   [Python 3.11+](https://www.python.org/downloads/)
*   [Visual Studio Code](https://code.visualstudio.com/)
*   API keys for [Regulations.gov](https://regulationsgov.github.io/developers/) and [Google AI Gemini](https://aistudio.google.com/app/apikey).

### Installation

1.  **Clone or Download the Project:**
    Save the project files into a local directory (e.g., `~/dev/regnavigator_lite`).

2.  **Open in VS Code:**
    ```bash
    cd ~/dev/regnavigator_lite
    code .
    ```

3.  **Create a Virtual Environment:**
    Open the terminal in VS Code (`View > Terminal`) and run the following commands:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    Your terminal prompt should now be prefixed with `(.venv)`.

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Up API Keys:**
    a. Create a `.env` file by copying the example file:
    ```bash
    cp .env.example .env
    ```
    b. Open the `.env` file in VS Code and replace the placeholder text with your actual API keys.

    ```dotenv
    # .env
    REGS_API_KEY="your_actual_regulations_gov_api_key"
    GEMINI_API_KEY="your_actual_google_ai_gemini_api_key"
    ```

## How to Run

With your virtual environment activated (`source .venv/bin/activate`), run the Streamlit app from the terminal:

```bash
streamlit run app.py