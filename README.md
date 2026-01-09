# Patient Health Report Metrics Extractor

## Overview
Patient Health Report Metrics Extractor is a professional, detail-oriented, reliable, empathetic healthcare agent designed for text interactions.

## Features


## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the agent:
```bash
python agent.py
```

## Configuration

The agent uses the following environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key (if using Anthropic)
- `GOOGLE_API_KEY`: Google API key (if using Google)

## Usage

```python
from agent import Patient Health Report Metrics ExtractorAgent

agent = Patient Health Report Metrics ExtractorAgent()
response = await agent.process_message("Hello!")
```

## Domain: healthcare
## Personality: professional, detail-oriented, reliable, empathetic
## Modality: text