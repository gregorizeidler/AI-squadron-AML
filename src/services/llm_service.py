"""
Large Language Model Service for AML Detection System

This service provides a unified interface for interacting with various
LLM providers (AWS Bedrock, OpenAI, Groq) for semantic analysis tasks.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, Union
from datetime import datetime

try:
    from langchain_aws import ChatBedrock
except ImportError:
    ChatBedrock = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from config.settings import settings


class LLMService:
    """
    Service for LLM interactions supporting multiple providers.
    
    Supports:
    - AWS Bedrock (Claude, Titan, etc.)
    - OpenAI (GPT-4, GPT-3.5)
    - Groq (Llama models)
    
    Provides both synchronous and asynchronous interfaces.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the LLM service.
        
        Args:
            provider: LLM provider to use ('bedrock', 'openai', 'groq')
                     If None, uses the provider from settings
        """
        self.provider = provider or settings.llm.provider
        self.logger = self._setup_logging()
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on provider"""
        try:
            if self.provider == "bedrock":
                self._initialize_bedrock()
            elif self.provider == "openai":
                self._initialize_openai()
            elif self.provider == "groq":
                self._initialize_groq()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
            self.logger.info(f"LLM service initialized with provider: {self.provider}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM service: {str(e)}")
            # Fallback to mock LLM for testing/development
            self._initialize_mock_llm()
    
    def _initialize_bedrock(self):
        """Initialize AWS Bedrock LLM"""
        if not ChatBedrock:
            raise ImportError("langchain-aws not installed. Install with: pip install langchain-aws")
        
        self.llm = ChatBedrock(
            model_id=settings.aws.bedrock_model_id,
            region_name=settings.aws.bedrock_region,
            model_kwargs={
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens or 4096
            }
        )
    
    def _initialize_openai(self):
        """Initialize OpenAI LLM"""
        if not ChatOpenAI:
            raise ImportError("langchain-openai not installed. Install with: pip install langchain-openai")
        
        if not settings.llm.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        self.llm = ChatOpenAI(
            model=settings.llm.openai_model,
            api_key=settings.llm.openai_api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.llm.timeout
        )
    
    def _initialize_groq(self):
        """Initialize Groq LLM"""
        if not ChatGroq:
            raise ImportError("langchain-groq not installed. Install with: pip install langchain-groq")
        
        if not settings.llm.groq_api_key:
            raise ValueError("Groq API key not configured")
        
        self.llm = ChatGroq(
            model=settings.llm.groq_model,
            api_key=settings.llm.groq_api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.llm.timeout,
            max_retries=settings.llm.max_retries
        )
    
    def _initialize_mock_llm(self):
        """Initialize mock LLM for testing/development"""
        self.llm = MockLLM()
        self.provider = "mock"
        self.logger.warning("Using mock LLM - for development/testing only")
    
    def analyze_text(
        self, 
        text: str, 
        analysis_type: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Synchronously analyze text using the LLM.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis being performed
            context: Additional context for the analysis
            
        Returns:
            LLM analysis response
        """
        try:
            start_time = datetime.utcnow()
            
            # Prepare the message
            message = self._prepare_message(text, analysis_type, context)
            
            # Get LLM response
            response = self.llm.invoke(message)
            
            # Extract content from response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Log the analysis
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._log_analysis(analysis_type, len(text), len(content), processing_time)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}")
            return f"Analysis unavailable due to error: {str(e)}"
    
    async def analyze_text_async(
        self, 
        text: str, 
        analysis_type: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Asynchronously analyze text using the LLM.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis being performed
            context: Additional context for the analysis
            
        Returns:
            LLM analysis response
        """
        try:
            start_time = datetime.utcnow()
            
            # Prepare the message
            message = self._prepare_message(text, analysis_type, context)
            
            # Get LLM response asynchronously
            response = await self.llm.ainvoke(message)
            
            # Extract content from response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Log the analysis
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._log_analysis(analysis_type, len(text), len(content), processing_time)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error in async LLM analysis: {str(e)}")
            return f"Analysis unavailable due to error: {str(e)}"
    
    def _prepare_message(
        self, 
        text: str, 
        analysis_type: str, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Prepare the message for LLM analysis"""
        
        # Base system prompt for AML analysis
        system_prompt = """You are an expert Anti-Money Laundering (AML) analyst with deep knowledge of:
- Money laundering techniques and patterns
- Financial crimes and suspicious activities
- Regulatory compliance requirements (BSA, FATF, etc.)
- Risk assessment methodologies
- Investigation procedures

Provide clear, actionable analysis focused on practical AML compliance needs.
Use specific risk indicators and evidence-based conclusions.
When identifying risks, use UPPERCASE for key risk factors."""
        
        # Analysis-specific prompts
        analysis_prompts = {
            "document_analysis": "Focus on trade-based money laundering, invoice manipulation, and document fraud indicators.",
            "crypto_analysis": "Focus on cryptocurrency-specific risks including mixers, privacy coins, and blockchain patterns.",
            "behavioral_analysis": "Focus on transaction patterns, structuring, velocity anomalies, and customer behavior changes.",
            "sanctions_screening": "Focus on sanctions compliance, entity screening, and politically exposed persons.",
            "geographic_risk": "Focus on jurisdictional risks, routing patterns, and cross-border compliance.",
            "enhanced_due_diligence": "Provide comprehensive risk assessment with specific investigation recommendations."
        }
        
        specific_prompt = analysis_prompts.get(analysis_type, "Provide general AML risk analysis.")
        
        # Construct the full message
        message = f"{system_prompt}\n\n{specific_prompt}\n\nAnalyze the following:\n\n{text}"
        
        # Add context if provided
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            message += f"\n\nAdditional Context:\n{context_str}"
        
        return message
    
    def _log_analysis(self, analysis_type: str, input_length: int, output_length: int, processing_time: float):
        """Log analysis metrics"""
        self.logger.info(
            f"LLM analysis completed",
            extra={
                "analysis_type": analysis_type,
                "provider": self.provider,
                "input_length": input_length,
                "output_length": output_length,
                "processing_time": processing_time
            }
        )
    
    def get_model_name(self) -> str:
        """Get the name of the current model"""
        if self.provider == "bedrock":
            return settings.aws.bedrock_model_id
        elif self.provider == "openai":
            return settings.llm.openai_model
        elif self.provider == "groq":
            return settings.llm.groq_model
        else:
            return f"{self.provider}_model"
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider"""
        return {
            "provider": self.provider,
            "model": self.get_model_name(),
            "temperature": settings.llm.temperature,
            "max_tokens": settings.llm.max_tokens,
            "timeout": settings.llm.timeout
        }
    
    def is_available(self) -> bool:
        """Check if the LLM service is available"""
        return self.llm is not None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup service-specific logging"""
        logger = logging.getLogger("aml_system.llm_service")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger


class MockLLM:
    """Mock LLM for development and testing"""
    
    def __init__(self):
        self.responses = {
            "document_analysis": """
            DOCUMENT ANALYSIS COMPLETE

            RISK INDICATORS IDENTIFIED:
            - INVOICE_MISMATCH: Price discrepancies detected
            - TRADE_BASED_LAUNDERING: Unusual pricing patterns
            - SHELL_COMPANY: Minimal business activity indicators

            RECOMMENDATIONS:
            - Enhanced documentation review required
            - Verify counterparty legitimacy
            - Cross-reference with customs data
            """,
            
            "crypto_analysis": """
            CRYPTOCURRENCY RISK ASSESSMENT

            HIGH RISK INDICATORS:
            - CRYPTO_MIXER_USAGE: Transaction involves mixing service
            - NEW_WALLET: Recently created wallet address
            - PRIVACY_COIN: Enhanced anonymity features

            COMPLIANCE CONCERNS:
            - Enhanced monitoring required
            - Source of funds verification needed
            - Regulatory reporting consideration
            """,
            
            "behavioral_analysis": """
            BEHAVIORAL PATTERN ANALYSIS

            SUSPICIOUS PATTERNS DETECTED:
            - STRUCTURING: Multiple sub-threshold transactions
            - VELOCITY_ANOMALY: Unusual transaction frequency
            - TIMING_PATTERN: Transactions at unusual hours

            RISK ASSESSMENT:
            - Pattern consistent with money laundering techniques
            - Requires immediate investigation
            - Enhanced monitoring recommended
            """,
            
            "enhanced_due_diligence": """
            ENHANCED DUE DILIGENCE REPORT

            EXECUTIVE SUMMARY:
            Multiple risk factors identified requiring enhanced investigation.

            KEY FINDINGS:
            - Customer profile inconsistencies detected
            - Transaction patterns raise AML concerns
            - Geographic exposure to high-risk jurisdictions

            RECOMMENDATIONS:
            - Continuous enhanced monitoring
            - Source of wealth verification
            - Regulatory consultation advised
            """
        }
    
    def invoke(self, message: str) -> MockResponse:
        """Mock synchronous invoke"""
        # Simple keyword matching to return appropriate response
        for analysis_type, response in self.responses.items():
            if analysis_type.replace("_", " ") in message.lower():
                return MockResponse(response)
        
        return MockResponse("MOCK ANALYSIS: No specific analysis pattern detected. Standard risk assessment applied.")
    
    async def ainvoke(self, message: str) -> MockResponse:
        """Mock asynchronous invoke"""
        # Simulate some processing time
        await asyncio.sleep(0.1)
        return self.invoke(message)


class MockResponse:
    """Mock response object"""
    
    def __init__(self, content: str):
        self.content = content