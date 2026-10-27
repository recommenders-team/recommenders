# Recommender of Recommenders

This tutorial demonstrates how to use OpenAI's reasoning models to create an AI-powered recommendation system advisor. The system analyzes business scenarios and suggests optimal recommender algorithms along with a detailed implementation plan.

## Overview

This project leverages OpenAI's reasoning models (particularly the "o3" model with "high" reasoning effort) to:

1. Analyze a given business scenario
2. Recommend the most suitable recommendation algorithms
3. Generate a comprehensive action plan for implementation

## Installation

### Prerequisites

- Python 3.8+
- Azure OpenAI Service credentials or OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/recommenders-team/recommenders.git
   cd recommenders/examples/07_tutorials/ai-reco-of-recos
   ```

2. Create a virtual environment (if you don't have one already):
    ```bash
    python -m venv .venv
    ```

    Activate the virtual environment:
    ```bash
    # On Windows
    .venv\Scripts\activate
    
    # On macOS/Linux
    source .venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, install with uv (faster Python package installer):
    ```bash
    pip install uv
    uv pip install -r requirements.txt
    ```
   

4. Set up environment variables:
   Rename the `env.sample` file to `.env`, and then fill in with your Azure OpenAI credentials:
   
   ```bash
   # Azure OpenAI Configuration
   AZURE_OPENAI_RESOURCE="your-azure-openai-resource"
   AZURE_OPENAI_KEY="your-azure-openai-key"
   AZURE_OPENAI_API_VERSION="2024-12-01-preview"
   
   # Model deployment names
   AZURE_OPENAI_MODEL_O3="o3"  # Required for this application, **rename to your own deployment**
   ```

   Alternatively, if using standard OpenAI:
   ```bash
   # OpenAI API Key
   OPENAI_API_KEY="your-openai-api-key"
   OPENAI_MODEL_O3="o3"
   ```

## Usage

### Jupyter Notebook

The easiest way to use this tool is through the provided Jupyter notebook:

```bash
jupyter notebook master_reco.ipynb
```

### Python Script

Alternatively, you can use the core functionality in your Python scripts:

```python
import os
from dotenv import load_dotenv
load_dotenv()

from utils.file_utils import *
from utils.openai_utils import *
from utils.openai_data_models import *
from master_reco.master_reco_models import *

# Initialize model info
model_info = TextProcessingModelnfo(
    model_name="o3",             
    reasoning_efforts="high",     
)

# Define your business scenario
scenario = """
A major task in applying recommendations in retail is to predict which products 
a user is most likely to engage with, based on their shopping history.
This scenario is commonly shown on the personalized home page.
"""

# Get recommender selection
reco_selection_prompt_file = locate_prompt("reco_selection.txt", ".")
reco_selection_template = read_file(reco_selection_prompt_file)
reco_selection_prompt = reco_selection_template.format(scenario=scenario)
reco_selection = call_llm_structured_outputs(
    reco_selection_prompt, 
    model_info=model_info,
    response_format=RankedAlgosResponse
)
print("Recommended algorithms:", reco_selection)

# Generate action plan
action_plan_prompt_file = locate_prompt("action_plan.txt", ".")
action_plan_template = read_file(action_plan_prompt_file)
action_plan_prompt = action_plan_template.format(
    scenario=scenario, 
    algos=str(reco_selection.model_dump())
)
plan_of_action = call_llm_structured_outputs(
    action_plan_prompt, 
    model_info=model_info,
    response_format=RecommendationDeploymentPlan
)
print("Implementation plan:", plan_of_action)
```

## How It Works

1. **Scenario Definition**: You provide a description of your recommendation system use case
2. **Algorithm Selection**: The LLM analyzes your scenario and selects up to 3 optimal recommender algorithms from a catalog of options
3. **Action Plan Generation**: The system creates a detailed implementation plan covering:
   - Candidate generators
   - Re-ranking strategies
   - Training infrastructure
   - Serving architecture
   - Evaluation metrics
   - Deployment considerations

## Example

### Input Scenario:

```
A major task in applying recommendations in retail is to predict which products a user is most likely to engage with or purchase, based on the shopping or viewing history of that user. This scenario is commonly shown on the personalized home page, feed or newsletter.
```

### Output:

1. **Recommended Algorithms**:
   - LightGCN (graph-based collaborative filtering)
   - ALS (matrix factorization)
   - LightGBM (gradient boosting)

2. **Implementation Plan**:
   - Detailed architecture for candidate generation
   - Model training specifications
   - Serving path on Azure
   - A/B testing strategy
   - Metrics to track
   - DevOps and monitoring considerations

## System Architecture

The system consists of several components:

- **Prompt Templates**: Pre-defined prompts that guide the LLM's reasoning
- **OpenAI Integration**: Utilities for interacting with OpenAI API
- **Data Models**: Pydantic models for structured outputs
- **Core Logic**: Processing pipeline to analyze scenarios and generate recommendations

## Benefits

- **Expert Knowledge**: Leverages advanced AI to provide expert-level recommendations
- **Structured Output**: Provides implementation-ready plans rather than just general advice
- **Customized Solutions**: Tailors recommendations to specific business scenarios
- **Comprehensive Coverage**: Addresses all aspects of deploying recommendation systems

## Credits

This project uses the [Recommenders](https://github.com/microsoft/recommenders) library as a reference for recommendation algorithms.

## License

MIT License