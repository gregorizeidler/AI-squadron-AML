"""
AML System Configuration Settings
"""
import os
from typing import Optional, List
from pydantic import BaseSettings, Field


class AWSSettings(BaseSettings):
    """AWS Configuration Settings"""
    region: str = Field(default="us-east-1", env="AWS_REGION")
    access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # Bedrock Configuration
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0", 
        env="BEDROCK_MODEL_ID"
    )
    bedrock_region: str = Field(default="us-east-1", env="BEDROCK_REGION")
    
    # DynamoDB Configuration
    dynamodb_table_prefix: str = Field(default="aml_system", env="DYNAMODB_TABLE_PREFIX")
    dynamodb_region: str = Field(default="us-east-1", env="DYNAMODB_REGION")
    
    # EventBridge Configuration
    eventbridge_source: str = Field(default="aml.detection.system", env="EVENTBRIDGE_SOURCE")
    eventbridge_detail_type: str = Field(default="AML Analysis Event", env="EVENTBRIDGE_DETAIL_TYPE")


class LLMSettings(BaseSettings):
    """LLM Provider Configuration"""
    provider: str = Field(default="bedrock", env="LLM_PROVIDER")  # bedrock, openai, groq
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    
    # Groq Configuration
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-70b-8192", env="GROQ_MODEL")
    
    # General LLM Settings
    temperature: float = Field(default=0.0, env="LLM_TEMPERATURE")
    max_tokens: Optional[int] = Field(default=None, env="LLM_MAX_TOKENS")
    timeout: Optional[int] = Field(default=60, env="LLM_TIMEOUT")
    max_retries: int = Field(default=3, env="LLM_MAX_RETRIES")


class RiskSettings(BaseSettings):
    """Risk Assessment Configuration"""
    high_risk_threshold: int = Field(default=75, env="HIGH_RISK_THRESHOLD")
    medium_risk_threshold: int = Field(default=45, env="MEDIUM_RISK_THRESHOLD")
    low_risk_threshold: int = Field(default=25, env="LOW_RISK_THRESHOLD")
    
    # Scoring Weights
    sanctions_hit_weight: int = Field(default=40, env="SANCTIONS_HIT_WEIGHT")
    pep_status_weight: int = Field(default=35, env="PEP_STATUS_WEIGHT")
    crypto_risks_weight: int = Field(default=25, env="CRYPTO_RISKS_WEIGHT")
    jurisdiction_risks_weight: int = Field(default=20, env="JURISDICTION_RISKS_WEIGHT")
    document_risks_weight: int = Field(default=15, env="DOCUMENT_RISKS_WEIGHT")
    behavioral_alerts_weight: int = Field(default=10, env="BEHAVIORAL_ALERTS_WEIGHT")


class SystemSettings(BaseSettings):
    """General System Configuration"""
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Performance Settings
    max_concurrent_analyses: int = Field(default=50, env="MAX_CONCURRENT_ANALYSES")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Rate Limiting
    api_rate_limit_per_minute: int = Field(default=1000, env="API_RATE_LIMIT_PER_MINUTE")
    external_api_rate_limit: int = Field(default=100, env="EXTERNAL_API_RATE_LIMIT")


class ExternalServicesSettings(BaseSettings):
    """External Services Configuration"""
    # Sanctions API
    sanctions_api_url: str = Field(
        default="https://api.sanctions-list.com/v1", 
        env="SANCTIONS_API_URL"
    )
    sanctions_api_key: Optional[str] = Field(default=None, env="SANCTIONS_API_KEY")
    
    # PEP API
    pep_api_url: str = Field(
        default="https://api.pep-database.com/v1", 
        env="PEP_API_URL"
    )
    pep_api_key: Optional[str] = Field(default=None, env="PEP_API_KEY")


class Settings(BaseSettings):
    """Main Settings Class"""
    aws: AWSSettings = AWSSettings()
    llm: LLMSettings = LLMSettings()
    risk: RiskSettings = RiskSettings()
    system: SystemSettings = SystemSettings()
    external: ExternalServicesSettings = ExternalServicesSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


# Risk Configuration Lists
HIGH_RISK_COUNTRIES = ["IR", "KP", "SY", "CU", "MM", "RU", "AF", "IQ"]
TAX_HAVENS = ["KY", "VG", "BM", "PA", "MT", "AE", "CH", "LU", "JE", "GG"]
SANCTIONED_ENTITIES = [
    "narcotics_cartel_xyz", 
    "terror_group_abc", 
    "sanctioned_russian_bank",
    "prohibited_organization",
    "blacklisted_entity"
]
DARKNET_MARKETS = [
    "AlphaMarket", 
    "Dark0d3", 
    "Hydra", 
    "SilkRoad3", 
    "DarkBazaar"
]

# Document Risk Patterns
DOCUMENT_RISK_PATTERNS = [
    "INVOICE_MISMATCH",
    "PHANTOM_SHIPMENT", 
    "PROHIBITED_GOODS",
    "SHELL_COMPANY",
    "DARKNET_CONNECTION",
    "TRADE_BASED_LAUNDERING",
    "UNUSUAL_PRICING",
    "DUPLICATE_INVOICING"
]

# Crypto Risk Indicators
CRYPTO_RISK_INDICATORS = [
    "CRYPTO_MIXER",
    "NEW_WALLET",
    "PRIVACY_COIN",
    "CHAIN_HOPPING",
    "TUMBLING_PATTERN",
    "EXCHANGE_SPLITTING",
    "DARKNET_CONNECTION"
]