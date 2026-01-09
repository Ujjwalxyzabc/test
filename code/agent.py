
import os
import re
import asyncio
from typing import Any, Dict, Optional, List, Tuple, Union
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator
from dotenv import load_dotenv
from loguru import logger
import openai
import httpx
from cryptography.fernet import Fernet
from oauthlib.oauth2 import BackendApplicationClient
from requests.auth import HTTPBasicAuth
import requests

# =========================
# Configuration Management
# =========================

class Config:
    """Configuration loader and validator."""
    load_dotenv()
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EHR_API_URL: str = os.getenv("EHR_API_URL", "")
    EHR_API_CLIENT_ID: str = os.getenv("EHR_API_CLIENT_ID", "")
    EHR_API_CLIENT_SECRET: str = os.getenv("EHR_API_CLIENT_SECRET", "")
    METRICS_VALIDATOR_API_URL: str = os.getenv("METRICS_VALIDATOR_API_URL", "")
    METRICS_VALIDATOR_CLIENT_ID: str = os.getenv("METRICS_VALIDATOR_CLIENT_ID", "")
    METRICS_VALIDATOR_CLIENT_SECRET: str = os.getenv("METRICS_VALIDATOR_CLIENT_SECRET", "")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    MAX_REPORT_LENGTH: int = 50000

    @classmethod
    def validate(cls):
        missing = []
        for attr in [
            "OPENAI_API_KEY", "EHR_API_URL", "EHR_API_CLIENT_ID", "EHR_API_CLIENT_SECRET",
            "METRICS_VALIDATOR_API_URL", "METRICS_VALIDATOR_CLIENT_ID", "METRICS_VALIDATOR_CLIENT_SECRET"
        ]:
            if not getattr(cls, attr):
                missing.append(attr)
        if missing:
            raise RuntimeError(f"Missing required configuration: {', '.join(missing)}")
        if not cls.ENCRYPTION_KEY:
            raise RuntimeError("Missing ENCRYPTION_KEY for data encryption.")

Config.validate()

# =========================
# Logging Configuration
# =========================

logger.add("agent_audit.log", rotation="10 MB", retention="30 days", level="INFO")

# =========================
# Pydantic Models
# =========================

class ReportInput(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=128)
    report_id: str = Field(..., min_length=1, max_length=128)
    report_text: str = Field(..., min_length=1, max_length=Config.MAX_REPORT_LENGTH)

    @field_validator("report_text")
    @classmethod
    def validate_report_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Report text cannot be empty.")
        if len(v) > Config.MAX_REPORT_LENGTH:
            raise ValueError(f"Report text exceeds maximum allowed length ({Config.MAX_REPORT_LENGTH} characters).")
        # Remove dangerous characters, excessive whitespace
        v = v.strip()
        v = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", v)
        return v

class StructuredMetricsOutput(BaseModel):
    success: bool
    metrics: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None
    tips: Optional[List[str]] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error_type: str
    description: str
    tips: Optional[List[str]] = None

# =========================
# Abstract Base Classes
# =========================

class InputHandlerBase:
    async def receive_report(self, input: ReportInput) -> str:
        raise NotImplementedError

    async def validate_format(self, input: ReportInput) -> bool:
        raise NotImplementedError

class ExtractorBase:
    async def extract_metrics(self, report_text: str) -> Dict[str, Any]:
        raise NotImplementedError

class ValidatorBase:
    async def validate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class PIIRedactorBase:
    def redact(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class EHRIntegratorBase:
    async def fetch_report(self, patient_id: str, report_id: str) -> Optional[str]:
        raise NotImplementedError

    async def store_metrics(self, patient_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class PromptManagerBase:
    def get_system_prompt(self) -> str:
        raise NotImplementedError

    def get_user_prompt_template(self) -> str:
        raise NotImplementedError

    def get_few_shot_examples(self) -> List[str]:
        raise NotImplementedError

class ErrorManagerBase:
    def handle_error(self, error_type: str, context: Optional[Dict[str, Any]] = None) -> ErrorResponse:
        raise NotImplementedError

    async def retry(self, operation, *args, **kwargs) -> Any:
        raise NotImplementedError

class AuditLoggerBase:
    def log(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError

    def retrieve_logs(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

class SecurityManagerBase:
    def encrypt(self, data: str) -> str:
        raise NotImplementedError

    def authenticate(self, user: str) -> bool:
        raise NotImplementedError

    def manage_session(self, user: str) -> str:
        raise NotImplementedError

    def purge_data(self, data_id: str) -> bool:
        raise NotImplementedError

# =========================
# Supporting Classes
# =========================

class SecurityManager(SecurityManagerBase):
    """Manages encryption, authentication, session, and data retention."""
    def __init__(self):
        self.fernet = Fernet(Config.ENCRYPTION_KEY.encode())

    def encrypt(self, data: str) -> str:
        return self.fernet.encrypt(data.encode()).decode()

    def authenticate(self, user: str) -> bool:
        # Placeholder for real authentication logic
        return True

    def manage_session(self, user: str) -> str:
        # Placeholder for session management
        return f"session_{user}"

    def purge_data(self, data_id: str) -> bool:
        # Placeholder for data purging logic
        logger.info(f"Purged data for ID: {data_id}")
        return True

class AuditLogger(AuditLoggerBase):
    """Logs all actions for compliance and traceability."""
    def log(self, event: Dict[str, Any]) -> None:
        logger.info(event)

    def retrieve_logs(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Placeholder: In production, query log storage
        return []

class ErrorManager(ErrorManagerBase):
    """Centralized error handling and fallback logic."""
    def handle_error(self, error_type: str, context: Optional[Dict[str, Any]] = None) -> ErrorResponse:
        description = f"Error occurred: {error_type}"
        tips = []
        if error_type == "MalformedJSON":
            description = "Malformed JSON in request. Please check for missing quotes, commas, or brackets."
            tips = [
                "Ensure all keys and string values are enclosed in double quotes.",
                "Check for missing or extra commas.",
                "Validate JSON structure before sending."
            ]
        elif error_type == "ValidationError":
            description = "Input validation failed. Please check the provided data."
            tips = [
                "Ensure patient_id and report_id are present and valid.",
                "Report text must not be empty and within allowed size."
            ]
        elif error_type == "LLMError":
            description = "LLM API error. Please try again later or contact support."
            tips = [
                "Check your API key and network connectivity.",
                "Try reducing input size if possible."
            ]
        elif error_type == "EHRAPIError":
            description = "EHR API error. Please try again later."
            tips = [
                "Check EHR API credentials and endpoint.",
                "Ensure patient_id and report_id are correct."
            ]
        elif error_type == "MetricsValidatorError":
            description = "Metrics Validator API error. Please try again later."
            tips = [
                "Check Metrics Validator API credentials and endpoint.",
                "Ensure metrics format is correct."
            ]
        else:
            description = f"Unknown error: {error_type}"
            tips = ["Contact support with error details."]
        logger.error({"error_type": error_type, "context": context, "description": description})
        return ErrorResponse(success=False, error_type=error_type, description=description, tips=tips)

    async def retry(self, operation, *args, **kwargs) -> Any:
        max_retries = 2
        delay = 1
        for attempt in range(max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Retry {attempt+1}/{max_retries} failed: {e}")
                await asyncio.sleep(delay)
                delay *= 2
        raise

class PromptManager(PromptManagerBase):
    """Manages system/user prompts and few-shot examples for LLM."""
    def get_system_prompt(self) -> str:
        return (
            "You are a professional healthcare agent. Your task is to read patient health reports and extract key clinical metrics such as vital signs, lab results, and medication lists. "
            "Output the results in a structured format. Do not provide medical advice or interpretation. Ensure all outputs comply with HIPAA and data privacy standards."
        )

    def get_user_prompt_template(self) -> str:
        return (
            "Please provide the patient health report text below. The agent will extract and summarize key metrics for your review."
        )

    def get_few_shot_examples(self) -> List[str]:
        return [
            "Patient: John Doe\nAge: 54\nBlood Pressure: 130/85 mmHg\nHeart Rate: 78 bpm\nLab Results: Glucose 110 mg/dL, Cholesterol 180 mg/dL\nMedications: Lisinopril 10mg daily",
            "Report: Female, 67 years old. Vitals: Temp 98.6°F, RR 16, BP 120/70. Labs: WBC 6.2, Hgb 13.2. No new medications."
        ]

class ReportInputHandler(InputHandlerBase):
    """Receives and pre-processes patient health report text."""
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager

    async def receive_report(self, input: ReportInput) -> str:
        # Authenticate user/session if needed
        if not self.security_manager.authenticate(input.patient_id):
            raise HTTPException(status_code=401, detail="Authentication failed.")
        return input.report_text

    async def validate_format(self, input: ReportInput) -> bool:
        # Already validated by Pydantic, but can add more checks
        if not input.report_text or not input.report_text.strip():
            return False
        if len(input.report_text) > Config.MAX_REPORT_LENGTH:
            return False
        return True

class MetricsExtractor(ExtractorBase):
    """Calls LLM to extract key clinical metrics from report text."""
    def __init__(self, prompt_manager: PromptManager, error_manager: ErrorManager):
        self.prompt_manager = prompt_manager
        self.error_manager = error_manager
        self.client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = "gpt-4o"
        self.fallback_model = "gpt-3.5-turbo"

    async def extract_metrics(self, report_text: str) -> Dict[str, Any]:
        system_prompt = self.prompt_manager.get_system_prompt()
        user_prompt = self.prompt_manager.get_user_prompt_template()
        few_shots = self.prompt_manager.get_few_shot_examples()
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        for example in few_shots:
            messages.append({"role": "user", "content": example})
        messages.append({"role": "user", "content": f"{user_prompt}\n\n{report_text}"})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1200
            )
            content = response.choices[0].message.content
            # Try to parse structured output (expecting JSON or key-value pairs)
            metrics = self._parse_metrics(content)
            return metrics
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            # Fallback to secondary model
            try:
                response = await self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1200
                )
                content = response.choices[0].message.content
                metrics = self._parse_metrics(content)
                return metrics
            except Exception as e2:
                logger.error(f"LLM fallback extraction error: {e2}")
                raise

    def _parse_metrics(self, content: str) -> Dict[str, Any]:
        # Try to extract JSON from LLM output
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                import json
                metrics = json.loads(match.group())
                return metrics
            # Fallback: parse key-value pairs
            metrics = {}
            for line in content.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metrics[key.strip()] = value.strip()
            return metrics
        except Exception as e:
            logger.warning(f"Failed to parse metrics: {e}")
            return {"raw_output": content}

class MetricsValidator(ValidatorBase):
    """Validates extracted metrics for accuracy and completeness."""
    def __init__(self, error_manager: ErrorManager):
        self.error_manager = error_manager

    async def validate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        # Call external Metrics Validator API
        token = await self._get_oauth_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.post(
                    Config.METRICS_VALIDATOR_API_URL,
                    json={"metrics": metrics},
                    headers=headers
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Metrics Validator API error: {response.text}")
                    raise Exception("MetricsValidatorError")
            except Exception as e:
                logger.error(f"Metrics Validator API call failed: {e}")
                raise

    async def _get_oauth_token(self) -> str:
        # OAuth2 Client Credentials Grant
        data = {
            "grant_type": "client_credentials",
            "client_id": Config.METRICS_VALIDATOR_CLIENT_ID,
            "client_secret": Config.METRICS_VALIDATOR_CLIENT_SECRET,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{Config.METRICS_VALIDATOR_API_URL}/oauth/token",
                data=data
            )
            if response.status_code == 200:
                return response.json().get("access_token", "")
            else:
                logger.error(f"Failed to get Metrics Validator OAuth token: {response.text}")
                raise Exception("MetricsValidatorError")

class PIIRedactor(PIIRedactorBase):
    """Redacts unnecessary PII from outputs."""
    def redact(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        # Remove keys that are not essential for clinical metrics
        pii_keys = ["Patient Name", "Name", "Address", "Phone", "SSN", "DOB", "Email"]
        redacted = {k: v for k, v in metrics.items() if k not in pii_keys}
        # Mask any detected PII patterns
        for k, v in redacted.items():
            if isinstance(v, str):
                v = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "***-**-****", v)  # SSN
                v = re.sub(r"\b\d{10}\b", "**********", v)  # Phone
                v = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "redacted_email", v)
                redacted[k] = v
        return redacted

class EHRIntegrator(EHRIntegratorBase):
    """Retrieves reports from and stores metrics to EHR systems."""
    def __init__(self, error_manager: ErrorManager):
        self.error_manager = error_manager

    async def fetch_report(self, patient_id: str, report_id: str) -> Optional[str]:
        token = await self._get_oauth_token()
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.get(
                    f"{Config.EHR_API_URL}/reports/{patient_id}/{report_id}",
                    headers=headers
                )
                if response.status_code == 200:
                    return response.json().get("report_text", "")
                else:
                    logger.error(f"EHR API fetch error: {response.text}")
                    raise Exception("EHRAPIError")
            except Exception as e:
                logger.error(f"EHR API fetch failed: {e}")
                raise

    async def store_metrics(self, patient_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        token = await self._get_oauth_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.post(
                    f"{Config.EHR_API_URL}/metrics/{patient_id}",
                    json={"metrics": metrics},
                    headers=headers
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"EHR API store error: {response.text}")
                    raise Exception("EHRAPIError")
            except Exception as e:
                logger.error(f"EHR API store failed: {e}")
                raise

    async def _get_oauth_token(self) -> str:
        # OAuth2 Client Credentials Grant
        data = {
            "grant_type": "client_credentials",
            "client_id": Config.EHR_API_CLIENT_ID,
            "client_secret": Config.EHR_API_CLIENT_SECRET,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{Config.EHR_API_URL}/oauth/token",
                data=data
            )
            if response.status_code == 200:
                return response.json().get("access_token", "")
            else:
                logger.error(f"Failed to get EHR OAuth token: {response.text}")
                raise Exception("EHRAPIError")

# =========================
# Main Agent Class
# =========================

class AgentBase:
    """Base class for all agents."""
    pass

class PatientHealthReportMetricsExtractorAgent(AgentBase):
    """
    Main agent orchestrating the extraction, validation, redaction, and output formatting.
    """
    def __init__(self):
        self.security_manager = SecurityManager()
        self.audit_logger = AuditLogger()
        self.error_manager = ErrorManager()
        self.prompt_manager = PromptManager()
        self.report_input_handler = ReportInputHandler(self.security_manager)
        self.metrics_extractor = MetricsExtractor(self.prompt_manager, self.error_manager)
        self.metrics_validator = MetricsValidator(self.error_manager)
        self.pii_redactor = PIIRedactor()
        self.ehr_integrator = EHRIntegrator(self.error_manager)

    async def process_report(
        self,
        patient_id: str,
        report_id: str,
        report_text: str
    ) -> StructuredMetricsOutput:
        """
        Orchestrates the end-to-end extraction, validation, redaction, and output formatting.
        """
        event_context = {
            "patient_id": patient_id,
            "report_id": report_id,
            "action": "process_report"
        }
        self.audit_logger.log({"event": "start_process_report", **event_context})
        try:
            # Input validation
            input_data = ReportInput(patient_id=patient_id, report_id=report_id, report_text=report_text)
            if not await self.report_input_handler.validate_format(input_data):
                raise ValidationError("Invalid report format.")
            self.audit_logger.log({"event": "input_validated", **event_context})

            # Extraction
            metrics = await self.error_manager.retry(self.metrics_extractor.extract_metrics, input_data.report_text)
            self.audit_logger.log({"event": "metrics_extracted", "metrics": metrics, **event_context})

            # Validation
            validation_result = await self.error_manager.retry(self.metrics_validator.validate, metrics)
            self.audit_logger.log({"event": "metrics_validated", "validation_result": validation_result, **event_context})

            if not validation_result.get("valid", True):
                raise Exception("MetricsValidatorError")

            # Redaction
            redacted_metrics = self.pii_redactor.redact(metrics)
            self.audit_logger.log({"event": "metrics_redacted", "redacted_metrics": redacted_metrics, **event_context})

            # Store metrics
            store_result = await self.error_manager.retry(self.ehr_integrator.store_metrics, patient_id, redacted_metrics)
            self.audit_logger.log({"event": "metrics_stored", "store_result": store_result, **event_context})

            return StructuredMetricsOutput(
                success=True,
                metrics=redacted_metrics,
                errors=None,
                tips=["Metrics extracted, validated, redacted, and stored successfully."]
            )
        except ValidationError as ve:
            error_resp = self.error_manager.handle_error("ValidationError", {"exception": str(ve), **event_context})
            return StructuredMetricsOutput(success=False, metrics=None, errors=[error_resp.model_dump()], tips=error_resp.tips)
        except Exception as e:
            error_type = getattr(e, "args", ["UnknownError"])[0]
            error_resp = self.error_manager.handle_error(error_type, {"exception": str(e), **event_context})
            return StructuredMetricsOutput(success=False, metrics=None, errors=[error_resp.model_dump()], tips=error_resp.tips)

    async def extract_metrics(self, report_text: str) -> Dict[str, Any]:
        """
        Extracts clinical metrics from report text using LLM.
        """
        try:
            metrics = await self.metrics_extractor.extract_metrics(report_text)
            return metrics
        except Exception as e:
            error_type = getattr(e, "args", ["LLMError"])[0]
            self.error_manager.handle_error(error_type, {"exception": str(e)})
            return {}

    async def validate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates extracted metrics for accuracy and completeness.
        """
        try:
            result = await self.metrics_validator.validate(metrics)
            return result
        except Exception as e:
            error_type = getattr(e, "args", ["MetricsValidatorError"])[0]
            self.error_manager.handle_error(error_type, {"exception": str(e)})
            return {}

    def redact_pii(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redacts unnecessary PII from metrics output.
        """
        try:
            return self.pii_redactor.redact(metrics)
        except Exception as e:
            self.error_manager.handle_error("RedactionError", {"exception": str(e)})
            return metrics

    async def store_metrics(self, patient_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stores validated, redacted metrics in EHR system.
        """
        try:
            result = await self.ehr_integrator.store_metrics(patient_id, metrics)
            return result
        except Exception as e:
            error_type = getattr(e, "args", ["EHRAPIError"])[0]
            self.error_manager.handle_error(error_type, {"exception": str(e)})
            return {}

    def handle_error(self, error_type: str, context: Optional[Dict[str, Any]] = None) -> ErrorResponse:
        """
        Centralized error handling and fallback logic.
        """
        return self.error_manager.handle_error(error_type, context)

# =========================
# FastAPI App & Endpoints
# =========================

app = FastAPI(
    title="Patient Health Report Metrics Extractor Agent",
    description="Extracts key clinical metrics from patient health reports using LLMs, validates, redacts PII, and stores results securely.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = PatientHealthReportMetricsExtractorAgent()

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    error_resp = agent.handle_error("ValidationError", {"exception": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_resp.model_dump()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_resp = agent.handle_error("HTTPException", {"exception": exc.detail})
    return JSONResponse(
        status_code=exc.status_code,
        content=error_resp.model_dump()
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    error_type = getattr(exc, "args", ["UnknownError"])[0]
    error_resp = agent.handle_error(error_type, {"exception": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_resp.model_dump()
    )

@app.post("/extract_metrics", response_model=StructuredMetricsOutput)
async def extract_metrics_endpoint(input: ReportInput):
    """
    Endpoint to extract, validate, redact, and store metrics from a patient health report.
    """
    try:
        # Input validation is handled by Pydantic
        result = await agent.process_report(
            patient_id=input.patient_id,
            report_id=input.report_id,
            report_text=input.report_text
        )
        return result
    except ValidationError as ve:
        error_resp = agent.handle_error("ValidationError", {"exception": str(ve)})
        return StructuredMetricsOutput(success=False, metrics=None, errors=[error_resp.model_dump()], tips=error_resp.tips)
    except Exception as e:
        error_type = getattr(e, "args", ["UnknownError"])[0]
        error_resp = agent.handle_error(error_type, {"exception": str(e)})
        return StructuredMetricsOutput(success=False, metrics=None, errors=[error_resp.model_dump()], tips=error_resp.tips)

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"success": True, "status": "ok"}

# =========================
# Main Execution Block
# =========================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Patient Health Report Metrics Extractor Agent...")
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)
