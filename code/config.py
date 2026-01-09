
# config.py

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class AgentConfig:
    # 1. Environment variable loading
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EHR_API_URL = os.getenv("EHR_API_URL")
    EHR_API_CLIENT_ID = os.getenv("EHR_API_CLIENT_ID")
    EHR_API_CLIENT_SECRET = os.getenv("EHR_API_CLIENT_SECRET")
    METRICS_VALIDATOR_API_URL = os.getenv("METRICS_VALIDATOR_API_URL")
    METRICS_VALIDATOR_CLIENT_ID = os.getenv("METRICS_VALIDATOR_CLIENT_ID")
    METRICS_VALIDATOR_CLIENT_SECRET = os.getenv("METRICS_VALIDATOR_CLIENT_SECRET")
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "default_dummy_key_for_dev")
    # 2. API key management
    API_KEYS = {
        "openai": OPENAI_API_KEY,
        "ehr": EHR_API_CLIENT_ID,
        "ehr_secret": EHR_API_CLIENT_SECRET,
        "metrics_validator": METRICS_VALIDATOR_CLIENT_ID,
        "metrics_validator_secret": METRICS_VALIDATOR_CLIENT_SECRET
    }
    # 3. LLM configuration
    LLM_CONFIG = {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 1200,
        "system_prompt": (
            "You are a professional healthcare agent. Your task is to read patient health reports and extract key clinical metrics such as vital signs, lab results, and medication lists. "
            "Output the results in a structured format. Do not provide medical advice or interpretation. Ensure all outputs comply with HIPAA and data privacy standards."
        ),
        "user_prompt_template": (
            "Please provide the patient health report text below. The agent will extract and summarize key metrics for your review."
        ),
        "few_shot_examples": [
            "Patient: John Doe\nAge: 54\nBlood Pressure: 130/85 mmHg\nHeart Rate: 78 bpm\nLab Results: Glucose 110 mg/dL, Cholesterol 180 mg/dL\nMedications: Lisinopril 10mg daily",
            "Report: Female, 67 years old. Vitals: Temp 98.6°F, RR 16, BP 120/70. Labs: WBC 6.2, Hgb 13.2. No new medications."
        ]
    }
    # 4. Domain-specific settings
    DOMAIN = "healthcare"
    AGENT_NAME = "Patient Health Report Metrics Extractor"
    API_REQUIREMENTS = [
        {
            "name": "EHR_API",
            "type": "external",
            "purpose": "Retrieve patient health reports and store extracted metrics.",
            "authentication": "OAuth 2.0 with multi-factor authentication",
            "rate_limits": "100 requests/minute per client"
        },
        {
            "name": "Metrics_Validator_API",
            "type": "external",
            "purpose": "Validate extracted metrics for accuracy and completeness.",
            "authentication": "OAuth 2.0",
            "rate_limits": "50 requests/minute per client"
        }
    ]
    # 5. Validation and error handling
    @classmethod
    def validate_config(cls):
        missing = []
        for key, value in cls.API_KEYS.items():
            if not value or value == "":
                missing.append(key)
        if missing:
            raise ConfigError(f"Missing required API keys: {', '.join(missing)}")
        if not cls.EHR_API_URL:
            raise ConfigError("Missing EHR_API_URL environment variable.")
        if not cls.METRICS_VALIDATOR_API_URL:
            raise ConfigError("Missing METRICS_VALIDATOR_API_URL environment variable.")
        if not cls.ENCRYPTION_KEY or cls.ENCRYPTION_KEY == "default_dummy_key_for_dev":
            # For production, enforce a real encryption key
            pass # comment: In dev, allow dummy key; in prod, raise error

    # 6. Default values and fallbacks
    FALLBACKS = {
        "llm_model": "gpt-3.5-turbo",
        "encryption_key": "default_dummy_key_for_dev"
    }
    # 7. Utility: Get config as dict
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        return {
            "agent_name": cls.AGENT_NAME,
            "domain": cls.DOMAIN,
            "llm_config": cls.LLM_CONFIG,
            "api_requirements": cls.API_REQUIREMENTS,
            "api_keys": cls.API_KEYS,
            "ehr_api_url": cls.EHR_API_URL,
            "metrics_validator_api_url": cls.METRICS_VALIDATOR_API_URL,
            "encryption_key": cls.ENCRYPTION_KEY,
            "fallbacks": cls.FALLBACKS
        }

# Validate config at import
try:
    AgentConfig.validate_config()
except ConfigError as e:
    # comment: In production, log and exit; here, print error for visibility
    print(f"CONFIG ERROR: {e}")

# comment: Example usage
# config = AgentConfig.as_dict()
