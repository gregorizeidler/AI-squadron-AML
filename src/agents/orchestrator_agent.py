"""
Orchestrator Agent for AML Detection System

This agent serves as the master coordinator, performing initial assessment
and guiding the overall workflow execution based on transaction characteristics.
"""
from typing import List
from decimal import Decimal

from .base_agent import BaseAgent
from ..core.state_manager import AMLState
from ..models.enums import AlertType, RiskLevel, AnalysisType
from ..services.llm_service import LLMService


class OrchestratorAgent(BaseAgent):
    """
    Master orchestrator agent that coordinates the AML detection workflow.
    
    Responsibilities:
    - Initial transaction assessment
    - Workflow routing decisions
    - High-level risk classification
    - Coordination between specialized agents
    """
    
    def __init__(self, llm_service: LLMService):
        """Initialize the orchestrator agent"""
        super().__init__(llm_service, "orchestrator")
    
    def analyze(self, state: AMLState) -> AMLState:
        """
        Perform comprehensive initial assessment of the transaction.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with initial assessment results
        """
        return self.initial_assessment(state)
    
    def initial_assessment(self, state: AMLState) -> AMLState:
        """
        Perform initial transaction assessment and classification.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with initial assessment
        """
        if not self.validate_state(state):
            return state
        
        self.log_analysis_start(state, "initial_assessment")
        
        # Update path tracking
        state = self.update_path(state, "initial_assessment")
        
        try:
            transaction = state["transaction"]
            customer = state["customer"]
            
            # Perform basic transaction classification
            state = self._classify_transaction(state)
            
            # Assess immediate red flags
            state = self._assess_immediate_red_flags(state)
            
            # Determine initial risk level
            state = self._determine_initial_risk_level(state)
            
            # Generate initial recommendations
            state = self._generate_initial_recommendations(state)
            
            self.log_analysis_complete(
                state, 
                "initial_assessment",
                findings_count=len(state["risk_factors"]),
                alerts_generated=len(state["alerts"])
            )
            
            return state
            
        except Exception as e:
            self.log_error(state, f"Error in initial assessment: {str(e)}", e)
            return state
    
    def _classify_transaction(self, state: AMLState) -> AMLState:
        """Classify transaction type and characteristics"""
        transaction = state["transaction"]
        customer = state["customer"]
        
        classification_factors = []
        
        # Amount-based classification
        if transaction.amount > Decimal("100000"):
            classification_factors.append("LARGE_TRANSACTION")
        elif transaction.amount > Decimal("10000"):
            classification_factors.append("REPORTABLE_AMOUNT")
        elif transaction.amount < Decimal("3000"):
            classification_factors.append("SMALL_TRANSACTION")
        
        # Asset type classification
        if transaction.asset_type.value == "CRYPTO":
            classification_factors.append("CRYPTOCURRENCY")
        elif transaction.asset_type.value == "PRECIOUS_METALS":
            classification_factors.append("PRECIOUS_METALS")
        
        # Customer classification
        if customer.account_age_days < 30:
            classification_factors.append("NEW_CUSTOMER")
        elif customer.account_age_days < 7:
            classification_factors.append("VERY_NEW_CUSTOMER")
        
        # Cross-border classification
        if transaction.origin_country != transaction.destination_country:
            classification_factors.append("CROSS_BORDER")
        
        if transaction.intermediate_countries:
            classification_factors.append("MULTI_HOP_TRANSACTION")
            if len(transaction.intermediate_countries) > 2:
                classification_factors.append("COMPLEX_ROUTING")
        
        # Add classification factors as metadata
        updated_state = state.copy()
        updated_state["metadata"]["transaction_classification"] = classification_factors
        
        return self.add_risk_factors(updated_state, classification_factors)
    
    def _assess_immediate_red_flags(self, state: AMLState) -> AMLState:
        """Assess immediate red flags that require urgent attention"""
        transaction = state["transaction"]
        customer = state["customer"]
        alerts_to_add = []
        
        # High-value new customer
        if (transaction.amount > Decimal("50000") and 
            customer.account_age_days < 7):
            alert = self.create_alert(
                AlertType.UNUSUAL_PATTERN,
                f"High-value transaction ({transaction.amount}) from very new customer (age: {customer.account_age_days} days)",
                severity=RiskLevel.HIGH,
                confidence=0.9,
                transaction_id=transaction.transaction_id,
                customer_id=customer.customer_id
            )
            alerts_to_add.append(alert)
        
        # Round number transactions
        if self._is_round_number(transaction.amount):
            alert = self.create_alert(
                AlertType.UNUSUAL_PATTERN,
                f"Round number transaction amount: {transaction.amount}",
                severity=RiskLevel.LOW,
                confidence=0.6,
                transaction_id=transaction.transaction_id,
                customer_id=customer.customer_id
            )
            alerts_to_add.append(alert)
        
        # Missing documentation for large transactions
        if (transaction.amount > Decimal("25000") and 
            len(transaction.documents) == 0):
            alert = self.create_alert(
                AlertType.MISSING_DOCUMENTS,
                f"Large transaction ({transaction.amount}) missing supporting documentation",
                severity=RiskLevel.MEDIUM,
                confidence=0.8,
                transaction_id=transaction.transaction_id,
                customer_id=customer.customer_id
            )
            alerts_to_add.append(alert)
        
        # Add all alerts to state
        updated_state = state
        for alert in alerts_to_add:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _determine_initial_risk_level(self, state: AMLState) -> AMLState:
        """Determine initial risk level based on immediate factors"""
        risk_factors = state["risk_factors"]
        alerts = state["alerts"]
        
        # Calculate initial risk score
        initial_score = 0
        
        # Score based on risk factors
        high_risk_factors = [
            "VERY_NEW_CUSTOMER", "LARGE_TRANSACTION", "COMPLEX_ROUTING",
            "CRYPTOCURRENCY", "PRECIOUS_METALS"
        ]
        
        for factor in risk_factors:
            if factor in high_risk_factors:
                initial_score += 15
            else:
                initial_score += 5
        
        # Score based on alerts
        for alert in alerts:
            if alert.severity == RiskLevel.CRITICAL:
                initial_score += 25
            elif alert.severity == RiskLevel.HIGH:
                initial_score += 15
            elif alert.severity == RiskLevel.MEDIUM:
                initial_score += 10
            else:
                initial_score += 5
        
        # Update state with initial risk score
        updated_state = state.copy()
        updated_state["risk_score"] = min(initial_score, 100)
        
        # Determine risk level
        if initial_score >= 60:
            updated_state["risk_level"] = RiskLevel.CRITICAL
        elif initial_score >= 40:
            updated_state["risk_level"] = RiskLevel.HIGH
        elif initial_score >= 20:
            updated_state["risk_level"] = RiskLevel.MEDIUM
        else:
            updated_state["risk_level"] = RiskLevel.LOW
        
        return updated_state
    
    def _generate_initial_recommendations(self, state: AMLState) -> AMLState:
        """Generate initial workflow recommendations"""
        transaction = state["transaction"]
        customer = state["customer"]
        risk_level = state["risk_level"]
        
        recommendations = []
        
        # Crypto-specific recommendations
        if transaction.asset_type.value == "CRYPTO":
            recommendations.append("CRYPTO_ENHANCED_SCREENING")
        
        # Cross-border recommendations
        if transaction.origin_country != transaction.destination_country:
            recommendations.append("GEOGRAPHIC_RISK_ASSESSMENT")
        
        # Customer-based recommendations
        if customer.account_age_days < 30:
            recommendations.append("ENHANCED_CUSTOMER_SCREENING")
        
        # Risk-based recommendations
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("IMMEDIATE_REVIEW_REQUIRED")
            recommendations.append("ENHANCED_DUE_DILIGENCE")
        
        # Document-based recommendations
        if transaction.documents:
            recommendations.append("DOCUMENT_ANALYSIS_REQUIRED")
        
        # Store recommendations in metadata
        updated_state = state.copy()
        updated_state["metadata"]["initial_recommendations"] = recommendations
        
        return updated_state
    
    def _is_round_number(self, amount: Decimal, tolerance: float = 0.01) -> bool:
        """
        Check if transaction amount is a round number.
        
        Args:
            amount: Transaction amount to check
            tolerance: Tolerance percentage for "round" numbers
            
        Returns:
            True if amount appears to be a round number
        """
        # Convert to float for analysis
        amount_float = float(amount)
        
        # Check for obvious round numbers
        round_bases = [1000, 5000, 10000, 25000, 50000, 100000]
        
        for base in round_bases:
            if amount_float % base == 0:
                return True
            
            # Check within tolerance
            remainder = amount_float % base
            if remainder <= base * tolerance or remainder >= base * (1 - tolerance):
                return True
        
        return False