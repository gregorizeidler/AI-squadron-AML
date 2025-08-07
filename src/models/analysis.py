"""
Analysis and investigation result models for the AML detection system
"""
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from .enums import RiskLevel, AlertType, InvestigationStatus, AnalysisType


class AlertDetails(BaseModel):
    """Details of an AML alert"""
    alert_id: str = Field(..., description="Unique alert identifier")
    alert_type: AlertType = Field(..., description="Type of alert")
    severity: RiskLevel = Field(..., description="Alert severity level")
    description: str = Field(..., description="Human-readable alert description")
    confidence_score: float = Field(..., description="Confidence in alert", ge=0, le=1)
    
    # Context information
    transaction_id: Optional[str] = Field(None, description="Related transaction ID")
    customer_id: Optional[str] = Field(None, description="Related customer ID")
    
    # Risk factors
    risk_factors: List[str] = Field(default_factory=list, description="Contributing risk factors")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    
    # Temporal information
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Alert creation time")
    expires_at: Optional[datetime] = Field(None, description="Alert expiration time")
    
    # Processing information
    auto_generated: bool = Field(True, description="Whether alert was auto-generated")
    requires_review: bool = Field(True, description="Whether human review is required")
    
    class Config:
        extra = "allow"


class RiskFactorDetail(BaseModel):
    """Detailed information about a risk factor"""
    factor_type: str = Field(..., description="Type of risk factor")
    factor_value: str = Field(..., description="Specific value or description")
    weight: int = Field(..., description="Weight in risk calculation")
    confidence: float = Field(..., description="Confidence in factor", ge=0, le=1)
    source: str = Field(..., description="Source of risk factor identification")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment results"""
    risk_score: int = Field(..., description="Overall risk score", ge=0, le=100)
    risk_level: RiskLevel = Field(..., description="Risk level classification")
    
    # Component scores
    geographic_risk: int = Field(default=0, description="Geographic risk component")
    transaction_risk: int = Field(default=0, description="Transaction-based risk component")
    customer_risk: int = Field(default=0, description="Customer-based risk component")
    behavioral_risk: int = Field(default=0, description="Behavioral risk component")
    document_risk: int = Field(default=0, description="Document-based risk component")
    
    # Risk factors
    risk_factors: List[RiskFactorDetail] = Field(
        default_factory=list, 
        description="Detailed risk factors"
    )
    
    # Assessment metadata
    assessment_date: datetime = Field(default_factory=datetime.utcnow)
    assessor: str = Field(default="SYSTEM", description="Who performed the assessment")
    assessment_version: str = Field(default="1.0", description="Assessment model version")
    
    # Recommendations
    recommended_action: str = Field(..., description="Recommended next action")
    requires_edd: bool = Field(False, description="Whether EDD is required")
    requires_sar: bool = Field(False, description="Whether SAR should be filed")
    
    class Config:
        extra = "allow"


class LLMAnalysisResult(BaseModel):
    """Results from LLM-powered analysis"""
    analysis_type: AnalysisType = Field(..., description="Type of LLM analysis")
    model_used: str = Field(..., description="LLM model identifier")
    
    # Analysis content
    analysis_text: str = Field(..., description="Full analysis text from LLM")
    key_findings: List[str] = Field(default_factory=list, description="Key findings extracted")
    risk_indicators: List[str] = Field(default_factory=list, description="Risk indicators identified")
    
    # Structured outputs
    extracted_entities: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Entities extracted from analysis"
    )
    sentiment_score: Optional[float] = Field(None, description="Sentiment score if applicable")
    confidence_score: float = Field(..., description="Overall confidence in analysis", ge=0, le=1)
    
    # Processing metadata
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        extra = "allow"


class DecisionRecord(BaseModel):
    """Record of an automated or manual decision"""
    decision_id: str = Field(..., description="Unique decision identifier")
    decision_type: str = Field(..., description="Type of decision made")
    decision: str = Field(..., description="The actual decision")
    
    # Decision context
    automated: bool = Field(True, description="Whether decision was automated")
    decision_maker: str = Field(..., description="Who or what made the decision")
    confidence: float = Field(..., description="Confidence in decision", ge=0, le=1)
    
    # Supporting information
    reasoning: str = Field(..., description="Reasoning behind the decision")
    supporting_evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    decision_factors: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Factors considered in decision"
    )
    
    # Temporal information
    decided_at: datetime = Field(default_factory=datetime.utcnow)
    effective_date: Optional[datetime] = Field(None, description="When decision takes effect")
    expiration_date: Optional[datetime] = Field(None, description="When decision expires")
    
    # Review information
    reviewed: bool = Field(False, description="Whether decision has been reviewed")
    reviewed_by: Optional[str] = Field(None, description="Who reviewed the decision")
    review_date: Optional[datetime] = Field(None, description="When decision was reviewed")
    review_outcome: Optional[str] = Field(None, description="Outcome of review")
    
    class Config:
        extra = "allow"


class AMLAnalysisResult(BaseModel):
    """Complete AML analysis result"""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    transaction_id: str = Field(..., description="Transaction being analyzed")
    customer_id: str = Field(..., description="Customer involved in transaction")
    
    # Analysis workflow
    decision_path: List[str] = Field(
        default_factory=list, 
        description="Path taken through analysis workflow"
    )
    workflow_version: str = Field(default="1.0", description="Version of workflow used")
    
    # Risk assessment
    risk_assessment: RiskAssessment = Field(..., description="Risk assessment results")
    
    # Alerts generated
    alerts: List[AlertDetails] = Field(default_factory=list, description="Alerts generated")
    
    # LLM analysis results
    llm_analyses: List[LLMAnalysisResult] = Field(
        default_factory=list, 
        description="LLM analysis results"
    )
    
    # Decisions and actions
    decisions: List[DecisionRecord] = Field(
        default_factory=list, 
        description="Decisions made during analysis"
    )
    
    # Final outcome
    final_status: str = Field(..., description="Final analysis status")
    requires_human_review: bool = Field(False, description="Whether human review is needed")
    sar_recommended: bool = Field(False, description="Whether SAR is recommended")
    transaction_approved: bool = Field(False, description="Whether transaction is approved")
    
    # Processing metadata
    processing_started: datetime = Field(default_factory=datetime.utcnow)
    processing_completed: Optional[datetime] = Field(None, description="When processing completed")
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time")
    
    # Review information
    reviewed_by: Optional[str] = Field(None, description="Human reviewer")
    review_date: Optional[datetime] = Field(None, description="Review date")
    review_notes: Optional[str] = Field(None, description="Review notes")
    
    # Case information
    case_id: Optional[str] = Field(None, description="Associated case ID if escalated")
    reporting_status: Optional[str] = Field(None, description="Regulatory reporting status")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('risk_assessment')
    def validate_risk_consistency(cls, v):
        """Ensure risk assessment is consistent"""
        if not (0 <= v.risk_score <= 100):
            raise ValueError("Risk score must be between 0 and 100")
        return v
    
    def get_highest_risk_alert(self) -> Optional[AlertDetails]:
        """Get the highest risk alert"""
        if not self.alerts:
            return None
        return max(self.alerts, key=lambda x: x.confidence_score)
    
    def get_primary_risk_factors(self) -> List[str]:
        """Get primary risk factors across all analyses"""
        factors = []
        for analysis in self.llm_analyses:
            factors.extend(analysis.risk_indicators)
        factors.extend([rf.factor_type for rf in self.risk_assessment.risk_factors])
        return list(set(factors))
    
    def is_high_risk(self) -> bool:
        """Check if analysis indicates high risk"""
        return self.risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }


class InvestigationRecord(BaseModel):
    """Record of an AML investigation case"""
    case_id: str = Field(..., description="Unique case identifier")
    case_type: str = Field(..., description="Type of investigation case")
    status: InvestigationStatus = Field(..., description="Current case status")
    priority: RiskLevel = Field(..., description="Case priority level")
    
    # Case details
    subject_customer_id: str = Field(..., description="Primary customer under investigation")
    related_transactions: List[str] = Field(
        default_factory=list, 
        description="Transaction IDs related to case"
    )
    related_customers: List[str] = Field(
        default_factory=list, 
        description="Related customer IDs"
    )
    
    # Investigation details
    allegations: List[str] = Field(default_factory=list, description="Allegations being investigated")
    evidence_collected: List[str] = Field(default_factory=list, description="Evidence collected")
    investigation_notes: List[str] = Field(default_factory=list, description="Investigation notes")
    
    # Analysis results
    analysis_results: List[str] = Field(
        default_factory=list, 
        description="IDs of related analysis results"
    )
    
    # Personnel
    assigned_investigator: Optional[str] = Field(None, description="Assigned investigator")
    reviewer: Optional[str] = Field(None, description="Case reviewer")
    
    # Temporal information
    opened_date: datetime = Field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = Field(None, description="Investigation due date")
    closed_date: Optional[datetime] = Field(None, description="Case closure date")
    
    # Outcome
    investigation_outcome: Optional[str] = Field(None, description="Investigation outcome")
    sar_filed: bool = Field(False, description="Whether SAR was filed")
    sar_number: Optional[str] = Field(None, description="SAR filing number")
    regulatory_action: Optional[str] = Field(None, description="Regulatory action taken")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional case metadata")
    
    def is_overdue(self) -> bool:
        """Check if investigation is overdue"""
        if not self.due_date:
            return False
        return datetime.utcnow() > self.due_date and self.status not in [
            InvestigationStatus.CLOSED_NO_ACTION,
            InvestigationStatus.CLOSED_SAR_FILED,
            InvestigationStatus.CLOSED_REFERRED
        ]
    
    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }