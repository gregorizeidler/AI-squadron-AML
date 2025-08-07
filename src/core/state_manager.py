"""
State Management for AML Analysis Workflow

This module manages the state throughout the LangGraph workflow execution,
ensuring proper data flow and state transitions.
"""
import hashlib
from datetime import datetime
from typing import TypedDict, List, Optional, Dict, Any
from decimal import Decimal

from ..models.transaction import Transaction
from ..models.customer import Customer
from ..models.analysis import (
    AlertDetails, RiskAssessment, LLMAnalysisResult, 
    DecisionRecord, AMLAnalysisResult
)
from ..models.enums import RiskLevel, TransactionStatus


class AMLState(TypedDict):
    """
    Typed dictionary representing the state of an AML analysis workflow.
    This state is passed between all nodes in the LangGraph workflow.
    """
    # Core entities
    transaction: Transaction
    customer: Customer
    
    # Analysis progress
    analysis_id: str
    decision_path: List[str]
    current_step: str
    workflow_version: str
    
    # Risk assessment
    risk_score: int
    risk_level: RiskLevel
    risk_factors: List[str]
    risk_assessment: Optional[RiskAssessment]
    
    # Alerts and findings
    alerts: List[AlertDetails]
    alert_count: int
    
    # Investigation data
    investigation: Dict[str, Any]
    llm_analysis: Dict[str, Any]
    llm_analyses: List[LLMAnalysisResult]
    
    # Documents and evidence
    documents: List[str]
    document_analysis_results: Dict[str, Any]
    
    # Compliance checks
    pep_status: Optional[bool]
    pep_category: Optional[str]
    sanction_hits: List[str]
    sanctions_checked: bool
    
    # Processing metadata
    case_id: Optional[str]
    transaction_count: int
    processing_started: datetime
    processing_time: float
    
    # Decision and status
    reporting_status: Optional[str]
    final_status: Optional[str]
    requires_human_review: bool
    sar_recommended: bool
    transaction_approved: bool
    
    # Additional context
    external_data: Dict[str, Any]
    metadata: Dict[str, Any]


class StateManager:
    """
    Manages AML workflow state transitions and ensures data integrity
    throughout the analysis process.
    """
    
    def __init__(self):
        """Initialize the state manager"""
        self.state_history: List[AMLState] = []
        self.checkpoints: Dict[str, AMLState] = {}
    
    def create_initial_state(
        self, 
        transaction: Transaction, 
        customer: Customer,
        workflow_version: str = "1.0"
    ) -> AMLState:
        """
        Create the initial state for a new AML analysis workflow.
        
        Args:
            transaction: The transaction to analyze
            customer: The customer associated with the transaction
            workflow_version: Version of the workflow being used
            
        Returns:
            Initial AMLState for the workflow
        """
        analysis_id = self._generate_analysis_id(transaction.transaction_id)
        
        initial_state: AMLState = {
            # Core entities
            "transaction": transaction,
            "customer": customer,
            
            # Analysis progress
            "analysis_id": analysis_id,
            "decision_path": [],
            "current_step": "initialization",
            "workflow_version": workflow_version,
            
            # Risk assessment
            "risk_score": 0,
            "risk_level": RiskLevel.LOW,
            "risk_factors": [],
            "risk_assessment": None,
            
            # Alerts and findings
            "alerts": [],
            "alert_count": 0,
            
            # Investigation data
            "investigation": {},
            "llm_analysis": {},
            "llm_analyses": [],
            
            # Documents and evidence
            "documents": [doc.document_content for doc in transaction.documents],
            "document_analysis_results": {},
            
            # Compliance checks
            "pep_status": None,
            "pep_category": None,
            "sanction_hits": [],
            "sanctions_checked": False,
            
            # Processing metadata
            "case_id": None,
            "transaction_count": 1,
            "processing_started": datetime.utcnow(),
            "processing_time": 0.0,
            
            # Decision and status
            "reporting_status": None,
            "final_status": None,
            "requires_human_review": False,
            "sar_recommended": False,
            "transaction_approved": False,
            
            # Additional context
            "external_data": {},
            "metadata": {}
        }
        
        return initial_state
    
    def update_path(self, state: AMLState, step: str) -> AMLState:
        """
        Update the decision path by adding a new step.
        
        Args:
            state: Current AML state
            step: Name of the step being entered
            
        Returns:
            Updated state with new step added to decision path
        """
        updated_state = state.copy()
        updated_state["decision_path"] = state["decision_path"] + [step]
        updated_state["current_step"] = step
        return updated_state
    
    def add_alert(self, state: AMLState, alert: AlertDetails) -> AMLState:
        """
        Add an alert to the state.
        
        Args:
            state: Current AML state
            alert: Alert to add
            
        Returns:
            Updated state with new alert
        """
        updated_state = state.copy()
        updated_state["alerts"] = state["alerts"] + [alert]
        updated_state["alert_count"] = len(updated_state["alerts"])
        return updated_state
    
    def add_risk_factor(self, state: AMLState, risk_factor: str) -> AMLState:
        """
        Add a risk factor to the state.
        
        Args:
            state: Current AML state
            risk_factor: Risk factor to add
            
        Returns:
            Updated state with new risk factor
        """
        updated_state = state.copy()
        if risk_factor not in state["risk_factors"]:
            updated_state["risk_factors"] = state["risk_factors"] + [risk_factor]
        return updated_state
    
    def update_risk_score(self, state: AMLState, score: int) -> AMLState:
        """
        Update the risk score and corresponding risk level.
        
        Args:
            state: Current AML state
            score: New risk score (0-100)
            
        Returns:
            Updated state with new risk score and level
        """
        updated_state = state.copy()
        updated_state["risk_score"] = max(0, min(100, score))  # Clamp to 0-100
        updated_state["risk_level"] = self._calculate_risk_level(score)
        return updated_state
    
    def add_llm_analysis(self, state: AMLState, analysis: LLMAnalysisResult) -> AMLState:
        """
        Add LLM analysis results to the state.
        
        Args:
            state: Current AML state
            analysis: LLM analysis result to add
            
        Returns:
            Updated state with new LLM analysis
        """
        updated_state = state.copy()
        updated_state["llm_analyses"] = state["llm_analyses"] + [analysis]
        
        # Also update the legacy llm_analysis dict for backwards compatibility
        analysis_type = analysis.analysis_type.value
        updated_state["llm_analysis"][analysis_type] = analysis.analysis_text
        
        return updated_state
    
    def set_final_status(
        self, 
        state: AMLState, 
        status: str,
        requires_review: bool = False,
        sar_recommended: bool = False,
        approved: bool = False
    ) -> AMLState:
        """
        Set the final status and decisions for the analysis.
        
        Args:
            state: Current AML state
            status: Final status string
            requires_review: Whether human review is required
            sar_recommended: Whether SAR is recommended
            approved: Whether transaction is approved
            
        Returns:
            Updated state with final status
        """
        updated_state = state.copy()
        updated_state["final_status"] = status
        updated_state["requires_human_review"] = requires_review
        updated_state["sar_recommended"] = sar_recommended
        updated_state["transaction_approved"] = approved
        
        # Update processing time
        processing_time = (datetime.utcnow() - state["processing_started"]).total_seconds()
        updated_state["processing_time"] = processing_time
        
        return updated_state
    
    def create_checkpoint(self, state: AMLState, checkpoint_name: str) -> None:
        """
        Create a checkpoint of the current state.
        
        Args:
            state: Current AML state
            checkpoint_name: Name for the checkpoint
        """
        self.checkpoints[checkpoint_name] = state.copy()
    
    def restore_checkpoint(self, checkpoint_name: str) -> Optional[AMLState]:
        """
        Restore a previously saved checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint to restore
            
        Returns:
            Restored state or None if checkpoint doesn't exist
        """
        return self.checkpoints.get(checkpoint_name)
    
    def get_state_summary(self, state: AMLState) -> Dict[str, Any]:
        """
        Get a summary of the current state for logging/monitoring.
        
        Args:
            state: Current AML state
            
        Returns:
            Dictionary containing state summary
        """
        return {
            "analysis_id": state["analysis_id"],
            "transaction_id": state["transaction"].transaction_id,
            "customer_id": state["customer"].customer_id,
            "current_step": state["current_step"],
            "decision_path": state["decision_path"],
            "risk_score": state["risk_score"],
            "risk_level": state["risk_level"].value,
            "alert_count": state["alert_count"],
            "requires_review": state["requires_human_review"],
            "sar_recommended": state["sar_recommended"],
            "processing_time": state["processing_time"]
        }
    
    def convert_to_analysis_result(self, state: AMLState) -> AMLAnalysisResult:
        """
        Convert the final state to an AMLAnalysisResult object.
        
        Args:
            state: Final AML state
            
        Returns:
            AMLAnalysisResult object containing all analysis results
        """
        return AMLAnalysisResult(
            analysis_id=state["analysis_id"],
            transaction_id=state["transaction"].transaction_id,
            customer_id=state["customer"].customer_id,
            decision_path=state["decision_path"],
            workflow_version=state["workflow_version"],
            risk_assessment=state["risk_assessment"] or RiskAssessment(
                risk_score=state["risk_score"],
                risk_level=state["risk_level"],
                recommended_action="UNKNOWN"
            ),
            alerts=state["alerts"],
            llm_analyses=state["llm_analyses"],
            final_status=state["final_status"] or "COMPLETED",
            requires_human_review=state["requires_human_review"],
            sar_recommended=state["sar_recommended"],
            transaction_approved=state["transaction_approved"],
            processing_started=state["processing_started"],
            processing_time_seconds=state["processing_time"],
            case_id=state["case_id"],
            reporting_status=state["reporting_status"],
            metadata=state["metadata"]
        )
    
    def _generate_analysis_id(self, transaction_id: str) -> str:
        """
        Generate a unique analysis ID based on transaction ID and timestamp.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            Unique analysis identifier
        """
        timestamp = datetime.utcnow().isoformat()
        content = f"{transaction_id}_{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _calculate_risk_level(self, score: int) -> RiskLevel:
        """
        Calculate risk level based on score.
        
        Args:
            score: Risk score (0-100)
            
        Returns:
            Corresponding risk level
        """
        if score >= 75:
            return RiskLevel.CRITICAL
        elif score >= 50:
            return RiskLevel.HIGH
        elif score >= 25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW