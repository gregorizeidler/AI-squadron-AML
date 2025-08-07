"""
Risk Calculation Engine for AML Detection System

This module handles the calculation of risk scores based on various factors
including sanctions, PEP status, geographic risks, behavioral patterns, etc.
"""
import yaml
from pathlib import Path
from typing import List, Dict, Any
from decimal import Decimal

from ..models.analysis import RiskAssessment, RiskFactorDetail
from ..models.enums import RiskLevel
from ..core.state_manager import AMLState
from config.settings import settings, HIGH_RISK_COUNTRIES, TAX_HAVENS


class RiskCalculator:
    """
    Calculates comprehensive risk scores for AML analysis using configurable
    weights and thresholds.
    """
    
    def __init__(self, config_path: str = "config/risk_parameters.yaml"):
        """
        Initialize the risk calculator with configuration parameters.
        
        Args:
            config_path: Path to risk parameters configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.scoring_weights = self.config.get("scoring_weights", {})
        self.risk_thresholds = self.config.get("risk_thresholds", {})
    
    def calculate_comprehensive_risk(self, state: AMLState) -> RiskAssessment:
        """
        Calculate comprehensive risk assessment for the given state.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Complete risk assessment with detailed breakdown
        """
        # Calculate component scores
        geographic_risk = self._calculate_geographic_risk(state)
        sanctions_risk = self._calculate_sanctions_risk(state)
        pep_risk = self._calculate_pep_risk(state)
        behavioral_risk = self._calculate_behavioral_risk(state)
        crypto_risk = self._calculate_crypto_risk(state)
        document_risk = self._calculate_document_risk(state)
        
        # Calculate overall risk score
        total_score = (
            geographic_risk +
            sanctions_risk +
            pep_risk +
            behavioral_risk +
            crypto_risk +
            document_risk
        )
        
        # Cap at maximum score
        total_score = min(total_score, 100)
        
        # Determine risk level
        risk_level = self._determine_risk_level(total_score)
        
        # Create detailed risk factors
        risk_factors = self._create_risk_factor_details(state, {
            "geographic": geographic_risk,
            "sanctions": sanctions_risk,
            "pep": pep_risk,
            "behavioral": behavioral_risk,
            "crypto": crypto_risk,
            "document": document_risk
        })
        
        # Determine recommendations
        recommended_action = self._determine_recommended_action(total_score, state)
        requires_edd = self._requires_enhanced_due_diligence(state, total_score)
        requires_sar = self._requires_sar_filing(total_score, state)
        
        return RiskAssessment(
            risk_score=total_score,
            risk_level=risk_level,
            geographic_risk=geographic_risk,
            transaction_risk=behavioral_risk,
            customer_risk=sanctions_risk + pep_risk,
            behavioral_risk=behavioral_risk,
            document_risk=document_risk,
            risk_factors=risk_factors,
            recommended_action=recommended_action,
            requires_edd=requires_edd,
            requires_sar=requires_sar
        )
    
    def _calculate_geographic_risk(self, state: AMLState) -> int:
        """Calculate risk score based on geographic factors"""
        score = 0
        transaction = state["transaction"]
        
        # High-risk countries
        high_risk_multiplier = self.config.get("geographic_risks", {}).get("high_risk_countries", {}).get("risk_multiplier", 2.0)
        
        if transaction.origin_country in HIGH_RISK_COUNTRIES:
            score += int(self.scoring_weights.get("jurisdiction_risks", 20) * high_risk_multiplier)
        
        if transaction.destination_country in HIGH_RISK_COUNTRIES:
            score += int(self.scoring_weights.get("jurisdiction_risks", 20) * high_risk_multiplier)
        
        # Check intermediate countries
        for country in transaction.intermediate_countries:
            if country in HIGH_RISK_COUNTRIES:
                score += int(self.scoring_weights.get("jurisdiction_risks", 20) * 0.5)
        
        # Tax havens
        tax_haven_multiplier = self.config.get("geographic_risks", {}).get("tax_havens", {}).get("risk_multiplier", 1.5)
        
        if transaction.origin_country in TAX_HAVENS:
            score += int(self.scoring_weights.get("jurisdiction_risks", 20) * tax_haven_multiplier)
        
        if transaction.destination_country in TAX_HAVENS:
            score += int(self.scoring_weights.get("jurisdiction_risks", 20) * tax_haven_multiplier)
        
        return min(score, self.scoring_weights.get("jurisdiction_risks", 20))
    
    def _calculate_sanctions_risk(self, state: AMLState) -> int:
        """Calculate risk score based on sanctions screening"""
        if state["sanction_hits"]:
            return self.scoring_weights.get("sanctions_hit", 40)
        return 0
    
    def _calculate_pep_risk(self, state: AMLState) -> int:
        """Calculate risk score based on PEP status"""
        if state["pep_status"]:
            return self.scoring_weights.get("pep_status", 35)
        return 0
    
    def _calculate_behavioral_risk(self, state: AMLState) -> int:
        """Calculate risk score based on behavioral patterns"""
        score = 0
        alerts = state["alerts"]
        
        # Count behavioral alerts
        behavioral_alert_types = [
            "STRUCTURING", "UNUSUAL_PATTERN", "RAPID_MOVEMENT"
        ]
        
        behavioral_alerts = [
            alert for alert in alerts 
            if any(alert_type in alert.alert_type.value for alert_type in behavioral_alert_types)
        ]
        
        score += len(behavioral_alerts) * self.scoring_weights.get("behavioral_alerts", 10)
        
        return min(score, self.scoring_weights.get("behavioral_alerts", 10) * 3)  # Cap at 3 alerts worth
    
    def _calculate_crypto_risk(self, state: AMLState) -> int:
        """Calculate risk score based on cryptocurrency-specific risks"""
        score = 0
        transaction = state["transaction"]
        
        if transaction.asset_type.value != "CRYPTO":
            return 0
        
        crypto_details = transaction.crypto_details
        if not crypto_details:
            return 0
        
        # Mixer usage
        if crypto_details.mixer_used:
            score += int(self.scoring_weights.get("crypto_risks", 25) * 0.8)
        
        # New wallet
        if crypto_details.wallet_age_days and crypto_details.wallet_age_days < 7:
            score += int(self.scoring_weights.get("crypto_risks", 25) * 0.3)
        
        # Privacy coin
        if crypto_details.privacy_coin:
            score += int(self.scoring_weights.get("crypto_risks", 25) * 0.5)
        
        # Multiple cross-chain swaps
        if crypto_details.cross_chain_swaps > 3:
            score += int(self.scoring_weights.get("crypto_risks", 25) * 0.4)
        
        # Darknet market connection
        if crypto_details.darknet_market:
            score += int(self.scoring_weights.get("crypto_risks", 25) * 0.9)
        
        return min(score, self.scoring_weights.get("crypto_risks", 25))
    
    def _calculate_document_risk(self, state: AMLState) -> int:
        """Calculate risk score based on document analysis"""
        score = 0
        
        # Check LLM analysis results for document risks
        for analysis in state["llm_analyses"]:
            if "document" in analysis.analysis_type.value.lower():
                # Count high-risk document indicators
                high_risk_indicators = [
                    "INVOICE_MISMATCH", "PROHIBITED_GOODS", "SHELL_COMPANY",
                    "TRADE_BASED_LAUNDERING", "PHANTOM_SHIPMENT"
                ]
                
                for indicator in analysis.risk_indicators:
                    if indicator in high_risk_indicators:
                        score += int(self.scoring_weights.get("document_risks", 15) * 0.3)
        
        return min(score, self.scoring_weights.get("document_risks", 15))
    
    def _determine_risk_level(self, score: int) -> RiskLevel:
        """Determine risk level based on score and thresholds"""
        if score >= self.risk_thresholds.get("high_risk", 75):
            return RiskLevel.CRITICAL
        elif score >= self.risk_thresholds.get("medium_risk", 45):
            return RiskLevel.HIGH
        elif score >= self.risk_thresholds.get("low_risk", 25):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _create_risk_factor_details(
        self, 
        state: AMLState, 
        component_scores: Dict[str, int]
    ) -> List[RiskFactorDetail]:
        """Create detailed risk factor breakdown"""
        factors = []
        
        for component, score in component_scores.items():
            if score > 0:
                factors.append(RiskFactorDetail(
                    factor_type=component.upper(),
                    factor_value=str(score),
                    weight=score,
                    confidence=0.9,  # High confidence in calculated scores
                    source="SYSTEM_CALCULATION",
                    evidence=self._get_evidence_for_component(state, component)
                ))
        
        # Add risk factors from state
        for risk_factor in state["risk_factors"]:
            factors.append(RiskFactorDetail(
                factor_type=risk_factor,
                factor_value="DETECTED",
                weight=5,  # Default weight for detected factors
                confidence=0.8,
                source="PATTERN_DETECTION",
                evidence=[f"Risk factor detected: {risk_factor}"]
            ))
        
        return factors
    
    def _get_evidence_for_component(self, state: AMLState, component: str) -> List[str]:
        """Get evidence supporting a risk component"""
        evidence = []
        
        if component == "geographic":
            transaction = state["transaction"]
            if transaction.origin_country in HIGH_RISK_COUNTRIES:
                evidence.append(f"Origin country {transaction.origin_country} is high-risk")
            if transaction.destination_country in HIGH_RISK_COUNTRIES:
                evidence.append(f"Destination country {transaction.destination_country} is high-risk")
            if transaction.origin_country in TAX_HAVENS:
                evidence.append(f"Origin country {transaction.origin_country} is a tax haven")
            if transaction.destination_country in TAX_HAVENS:
                evidence.append(f"Destination country {transaction.destination_country} is a tax haven")
        
        elif component == "sanctions":
            if state["sanction_hits"]:
                evidence.extend([f"Sanctions hit: {hit}" for hit in state["sanction_hits"]])
        
        elif component == "pep":
            if state["pep_status"]:
                evidence.append(f"Customer is a PEP: {state.get('pep_category', 'Unknown category')}")
        
        elif component == "behavioral":
            evidence.extend([f"Alert: {alert.description}" for alert in state["alerts"]])
        
        elif component == "crypto":
            crypto_details = state["transaction"].crypto_details
            if crypto_details:
                if crypto_details.mixer_used:
                    evidence.append("Cryptocurrency mixer used")
                if crypto_details.wallet_age_days and crypto_details.wallet_age_days < 7:
                    evidence.append(f"New wallet (age: {crypto_details.wallet_age_days} days)")
                if crypto_details.darknet_market:
                    evidence.append(f"Connected to darknet market: {crypto_details.darknet_market}")
        
        elif component == "document":
            for analysis in state["llm_analyses"]:
                if "document" in analysis.analysis_type.value.lower():
                    evidence.extend(analysis.risk_indicators)
        
        return evidence
    
    def _determine_recommended_action(self, score: int, state: AMLState) -> str:
        """Determine recommended action based on risk score"""
        if score >= self.risk_thresholds.get("high_risk", 75):
            return "FILE_SAR_IMMEDIATELY"
        elif score >= self.risk_thresholds.get("medium_risk", 45):
            return "HUMAN_REVIEW_REQUIRED"
        elif score >= self.risk_thresholds.get("low_risk", 25):
            return "ENHANCED_MONITORING"
        else:
            return "STANDARD_PROCESSING"
    
    def _requires_enhanced_due_diligence(self, state: AMLState, score: int) -> bool:
        """Determine if enhanced due diligence is required"""
        return (
            score >= self.risk_thresholds.get("medium_risk", 45) or
            state["pep_status"] or
            bool(state["sanction_hits"]) or
            state["transaction"].asset_type.value == "CRYPTO"
        )
    
    def _requires_sar_filing(self, score: int, state: AMLState) -> bool:
        """Determine if SAR filing is recommended"""
        return (
            score >= self.risk_thresholds.get("high_risk", 75) or
            bool(state["sanction_hits"])
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load risk calculation configuration from YAML file"""
        if not self.config_path.exists():
            # Return default configuration if file doesn't exist
            return {
                "scoring_weights": {
                    "sanctions_hit": 40,
                    "pep_status": 35,
                    "crypto_risks": 25,
                    "jurisdiction_risks": 20,
                    "document_risks": 15,
                    "behavioral_alerts": 10
                },
                "risk_thresholds": {
                    "high_risk": 75,
                    "medium_risk": 45,
                    "low_risk": 25
                }
            }
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading risk configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default risk calculation configuration"""
        return {
            "scoring_weights": {
                "sanctions_hit": 40,
                "pep_status": 35,
                "crypto_risks": 25,
                "jurisdiction_risks": 20,
                "document_risks": 15,
                "behavioral_alerts": 10
            },
            "risk_thresholds": {
                "high_risk": 75,
                "medium_risk": 45,
                "low_risk": 25
            }
        }