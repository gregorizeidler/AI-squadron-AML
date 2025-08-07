"""
Base Agent Class for AML Detection System

This module provides the base class for all specialized AML detection agents,
ensuring consistent behavior and providing common functionality.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..core.state_manager import AMLState
from ..models.analysis import AlertDetails, LLMAnalysisResult
from ..models.enums import AlertType, RiskLevel, AnalysisType
from ..services.llm_service import LLMService


class BaseAgent(ABC):
    """
    Abstract base class for all AML detection agents.
    
    Provides common functionality and enforces consistent interface
    across all specialized agents in the system.
    """
    
    def __init__(self, llm_service: LLMService, agent_name: str):
        """
        Initialize the base agent.
        
        Args:
            llm_service: Service for LLM interactions
            agent_name: Name of the specific agent
        """
        self.llm_service = llm_service
        self.agent_name = agent_name
        self.logger = self._setup_logging()
        
        # Agent configuration
        self.confidence_threshold = 0.7
        self.max_retries = 3
        self.timeout_seconds = 60
    
    @abstractmethod
    def analyze(self, state: AMLState) -> AMLState:
        """
        Main analysis method that each agent must implement.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with agent's analysis results
        """
        pass
    
    def update_path(self, state: AMLState, step: str) -> AMLState:
        """
        Update the decision path to track agent execution.
        
        Args:
            state: Current state
            step: Step name to add to path
            
        Returns:
            Updated state with new step in decision path
        """
        updated_state = state.copy()
        updated_state["decision_path"] = state["decision_path"] + [f"{self.agent_name}_{step}"]
        return updated_state
    
    def create_alert(
        self,
        alert_type: AlertType,
        description: str,
        severity: RiskLevel = RiskLevel.MEDIUM,
        confidence: float = 0.8,
        evidence: Optional[List[str]] = None,
        transaction_id: Optional[str] = None,
        customer_id: Optional[str] = None
    ) -> AlertDetails:
        """
        Create a standardized alert.
        
        Args:
            alert_type: Type of alert being created
            description: Human-readable description
            severity: Alert severity level
            confidence: Confidence in the alert (0-1)
            evidence: Supporting evidence list
            transaction_id: Related transaction ID
            customer_id: Related customer ID
            
        Returns:
            Structured alert details
        """
        import uuid
        
        return AlertDetails(
            alert_id=f"{self.agent_name}_{uuid.uuid4().hex[:8]}",
            alert_type=alert_type,
            severity=severity,
            description=description,
            confidence_score=confidence,
            transaction_id=transaction_id,
            customer_id=customer_id,
            evidence=evidence or [],
            created_at=datetime.utcnow(),
            auto_generated=True,
            requires_review=severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        )
    
    def add_alert_to_state(self, state: AMLState, alert: AlertDetails) -> AMLState:
        """
        Add an alert to the analysis state.
        
        Args:
            state: Current state
            alert: Alert to add
            
        Returns:
            Updated state with new alert
        """
        updated_state = state.copy()
        updated_state["alerts"] = state["alerts"] + [alert]
        updated_state["alert_count"] = len(updated_state["alerts"])
        return updated_state
    
    def add_risk_factors(self, state: AMLState, risk_factors: List[str]) -> AMLState:
        """
        Add risk factors to the analysis state.
        
        Args:
            state: Current state
            risk_factors: List of risk factors to add
            
        Returns:
            Updated state with new risk factors
        """
        updated_state = state.copy()
        existing_factors = set(state["risk_factors"])
        new_factors = [rf for rf in risk_factors if rf not in existing_factors]
        
        if new_factors:
            updated_state["risk_factors"] = state["risk_factors"] + new_factors
        
        return updated_state
    
    def create_llm_analysis_result(
        self,
        analysis_type: AnalysisType,
        analysis_text: str,
        key_findings: List[str],
        risk_indicators: List[str],
        confidence_score: float,
        model_used: str = "default",
        extracted_entities: Optional[Dict[str, Any]] = None,
        tokens_used: Optional[int] = None,
        processing_time: Optional[float] = None
    ) -> LLMAnalysisResult:
        """
        Create a standardized LLM analysis result.
        
        Args:
            analysis_type: Type of analysis performed
            analysis_text: Full text of the analysis
            key_findings: Key findings extracted
            risk_indicators: Risk indicators identified
            confidence_score: Confidence in analysis
            model_used: LLM model identifier
            extracted_entities: Entities extracted from analysis
            tokens_used: Number of tokens consumed
            processing_time: Time taken for analysis
            
        Returns:
            Structured LLM analysis result
        """
        return LLMAnalysisResult(
            analysis_type=analysis_type,
            model_used=model_used,
            analysis_text=analysis_text,
            key_findings=key_findings,
            risk_indicators=risk_indicators,
            extracted_entities=extracted_entities or {},
            confidence_score=confidence_score,
            tokens_used=tokens_used,
            processing_time=processing_time,
            created_at=datetime.utcnow()
        )
    
    def add_llm_analysis_to_state(self, state: AMLState, analysis: LLMAnalysisResult) -> AMLState:
        """
        Add LLM analysis result to the state.
        
        Args:
            state: Current state
            analysis: LLM analysis result to add
            
        Returns:
            Updated state with new analysis
        """
        updated_state = state.copy()
        updated_state["llm_analyses"] = state["llm_analyses"] + [analysis]
        
        # Also update the legacy llm_analysis dict for backwards compatibility
        analysis_key = f"{self.agent_name}_{analysis.analysis_type.value.lower()}"
        updated_state["llm_analysis"][analysis_key] = analysis.analysis_text
        
        return updated_state
    
    def log_analysis_start(self, state: AMLState, analysis_description: str) -> None:
        """
        Log the start of an analysis process.
        
        Args:
            state: Current state
            analysis_description: Description of the analysis being performed
        """
        self.logger.info(
            f"{self.agent_name} starting {analysis_description}",
            extra={
                "agent": self.agent_name,
                "analysis_id": state["analysis_id"],
                "transaction_id": state["transaction"].transaction_id,
                "customer_id": state["customer"].customer_id,
                "step": analysis_description
            }
        )
    
    def log_analysis_complete(
        self,
        state: AMLState,
        analysis_description: str,
        findings_count: int = 0,
        alerts_generated: int = 0
    ) -> None:
        """
        Log the completion of an analysis process.
        
        Args:
            state: Current state
            analysis_description: Description of the analysis performed
            findings_count: Number of findings generated
            alerts_generated: Number of alerts generated
        """
        self.logger.info(
            f"{self.agent_name} completed {analysis_description}",
            extra={
                "agent": self.agent_name,
                "analysis_id": state["analysis_id"],
                "transaction_id": state["transaction"].transaction_id,
                "customer_id": state["customer"].customer_id,
                "step": analysis_description,
                "findings_count": findings_count,
                "alerts_generated": alerts_generated
            }
        )
    
    def log_error(self, state: AMLState, error_message: str, exception: Optional[Exception] = None) -> None:
        """
        Log an error during analysis.
        
        Args:
            state: Current state
            error_message: Error description
            exception: Exception object if available
        """
        self.logger.error(
            f"{self.agent_name} error: {error_message}",
            extra={
                "agent": self.agent_name,
                "analysis_id": state["analysis_id"],
                "transaction_id": state["transaction"].transaction_id,
                "customer_id": state["customer"].customer_id,
                "error": error_message
            },
            exc_info=exception
        )
    
    def validate_state(self, state: AMLState) -> bool:
        """
        Validate that the state contains required information for this agent.
        
        Args:
            state: State to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        required_keys = ["transaction", "customer", "analysis_id"]
        
        for key in required_keys:
            if key not in state:
                self.logger.error(f"Missing required state key: {key}")
                return False
        
        return True
    
    def extract_risk_indicators_from_text(self, text: str) -> List[str]:
        """
        Extract risk indicators from text using pattern matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of identified risk indicators
        """
        import re
        
        # Common risk indicator patterns
        risk_patterns = [
            r'\b[A-Z_]{4,}\b',  # All caps words (likely risk codes)
            r'HIGH[\s_]RISK',
            r'SUSPICIOUS',
            r'SHELL[\s_]COMPANY',
            r'STRUCTURING',
            r'MIXER',
            r'DARKNET',
            r'SANCTIONS',
            r'PEP',
            r'MONEY[\s_]LAUNDERING'
        ]
        
        indicators = []
        for pattern in risk_patterns:
            matches = re.findall(pattern, text.upper())
            indicators.extend(matches)
        
        # Remove duplicates and return
        return list(set(indicators))
    
    def _setup_logging(self) -> logging.Logger:
        """Setup agent-specific logging"""
        logger = logging.getLogger(f"aml_agent.{self.agent_name}")
        logger.setLevel(logging.INFO)
        
        # Create handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger