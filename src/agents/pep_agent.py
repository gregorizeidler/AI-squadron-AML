"""
PEP (Politically Exposed Person) Screening Agent for AML Detection System

This agent specializes in identifying and assessing risks associated with
Politically Exposed Persons and their associates.
"""
from typing import List, Dict, Set
import re

from .base_agent import BaseAgent
from ..core.state_manager import AMLState
from ..models.enums import AlertType, RiskLevel, AnalysisType
from ..services.llm_service import LLMService


class PEPScreeningAgent(BaseAgent):
    """
    Specialized agent for Politically Exposed Person (PEP) screening and assessment.
    
    Responsibilities:
    - Identify PEPs among customers and beneficial owners
    - Classify PEP categories and risk levels
    - Screen for PEP associates and family members
    - Assess enhanced due diligence requirements
    - Monitor for changes in PEP status
    """
    
    def __init__(self, llm_service: LLMService):
        """Initialize the PEP screening agent"""
        super().__init__(llm_service, "pep_screener")
        
        # PEP category keywords for identification
        self.pep_categories = {
            "senior_political_figures": {
                "keywords": [
                    "president", "prime minister", "minister", "secretary",
                    "governor", "mayor", "senator", "congressman", "mp",
                    "deputy", "assistant secretary", "undersecretary"
                ],
                "risk_level": RiskLevel.CRITICAL,
                "confidence": 0.9
            },
            "judicial_officials": {
                "keywords": [
                    "judge", "justice", "chief justice", "prosecutor",
                    "attorney general", "district attorney", "magistrate"
                ],
                "risk_level": RiskLevel.HIGH,
                "confidence": 0.85
            },
            "military_officials": {
                "keywords": [
                    "general", "admiral", "colonel", "commander",
                    "major general", "brigadier", "marshal"
                ],
                "risk_level": RiskLevel.HIGH,
                "confidence": 0.8
            },
            "senior_government_officials": {
                "keywords": [
                    "director", "commissioner", "bureau chief",
                    "department head", "agency head", "ambassador",
                    "consul", "diplomatic"
                ],
                "risk_level": RiskLevel.MEDIUM,
                "confidence": 0.75
            },
            "political_party_officials": {
                "keywords": [
                    "party chairman", "party leader", "party secretary",
                    "party treasurer", "political party"
                ],
                "risk_level": RiskLevel.MEDIUM,
                "confidence": 0.7
            },
            "international_organization": {
                "keywords": [
                    "un ", "united nations", "world bank", "imf",
                    "international monetary", "european union",
                    "african union", "nato"
                ],
                "risk_level": RiskLevel.MEDIUM,
                "confidence": 0.8
            }
        }
        
        # Family relationship indicators
        self.family_indicators = [
            "spouse", "wife", "husband", "son", "daughter",
            "father", "mother", "brother", "sister",
            "son-in-law", "daughter-in-law", "parent",
            "child", "sibling", "relative"
        ]
        
        # Business associate indicators
        self.associate_indicators = [
            "business partner", "close associate", "advisor",
            "consultant", "agent", "representative",
            "nominee", "proxy", "front person"
        ]
    
    def analyze(self, state: AMLState) -> AMLState:
        """
        Perform comprehensive PEP screening analysis.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with PEP screening results
        """
        return self.screen_pep(state)
    
    def screen_pep(self, state: AMLState) -> AMLState:
        """
        Perform PEP screening for customer and related parties.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with PEP screening results
        """
        if not self.validate_state(state):
            return state
        
        self.log_analysis_start(state, "pep_screening")
        
        # Update path tracking
        state = self.update_path(state, "pep_screening")
        
        try:
            customer = state["customer"]
            transaction = state["transaction"]
            
            # Screen primary customer
            customer_pep_result = self._screen_individual(customer.name)
            
            # Screen beneficial owners
            beneficial_owner_results = []
            for owner in customer.beneficial_owners:
                result = self._screen_individual(owner.owner_name)
                if result["is_pep"]:
                    beneficial_owner_results.append((owner, result))
            
            # Screen transaction parties
            party_results = []
            for party in transaction.parties:
                result = self._screen_individual(party)
                if result["is_pep"]:
                    party_results.append((party, result))
            
            # Update state with PEP findings
            state = self._update_state_with_pep_results(
                state, customer_pep_result, beneficial_owner_results, party_results
            )
            
            # Generate PEP-related alerts
            state = self._generate_pep_alerts(
                state, customer_pep_result, beneficial_owner_results, party_results
            )
            
            # Assess enhanced due diligence requirements
            state = self._assess_edd_requirements(state)
            
            self.log_analysis_complete(
                state,
                "pep_screening",
                findings_count=int(state["pep_status"] or False) + len(beneficial_owner_results) + len(party_results),
                alerts_generated=len([a for a in state["alerts"] if "PEP" in a.description])
            )
            
            return state
            
        except Exception as e:
            self.log_error(state, f"Error in PEP screening: {str(e)}", e)
            return state
    
    def _screen_individual(self, name: str) -> Dict[str, any]:
        """
        Screen an individual name for PEP indicators.
        
        Args:
            name: Name to screen
            
        Returns:
            Dictionary with screening results
        """
        result = {
            "name": name,
            "is_pep": False,
            "pep_category": None,
            "confidence": 0.0,
            "matches": [],
            "risk_level": RiskLevel.LOW
        }
        
        if not name or not isinstance(name, str):
            return result
        
        name_lower = name.lower()
        
        # Check against PEP category keywords
        for category, config in self.pep_categories.items():
            for keyword in config["keywords"]:
                if keyword in name_lower:
                    result["is_pep"] = True
                    result["pep_category"] = category
                    result["confidence"] = config["confidence"]
                    result["risk_level"] = config["risk_level"]
                    result["matches"].append(keyword)
                    break
            
            if result["is_pep"]:
                break
        
        # Enhanced screening for potential variants
        if not result["is_pep"]:
            result = self._enhanced_name_screening(name_lower, result)
        
        return result
    
    def _enhanced_name_screening(self, name_lower: str, result: Dict) -> Dict:
        """
        Perform enhanced screening for name variations and titles.
        
        Args:
            name_lower: Lowercase name to screen
            result: Initial screening result
            
        Returns:
            Updated screening result
        """
        # Look for common title patterns
        title_patterns = [
            r'\b(hon|honorable|rt\s+hon|his\s+excellency|her\s+excellency)\b',
            r'\b(former|ex-|retired)\s+(president|minister|governor|mayor)\b',
            r'\b(chief|deputy|assistant|associate)\s+(minister|secretary|commissioner)\b'
        ]
        
        for pattern in title_patterns:
            if re.search(pattern, name_lower):
                result["is_pep"] = True
                result["pep_category"] = "government_official"
                result["confidence"] = 0.7
                result["risk_level"] = RiskLevel.MEDIUM
                result["matches"].append(f"Title pattern: {pattern}")
                break
        
        # Check for government department affiliations
        govt_indicators = [
            "ministry", "department of", "office of", "bureau of",
            "commission", "agency", "administration", "authority"
        ]
        
        for indicator in govt_indicators:
            if indicator in name_lower:
                # Lower confidence as this could be employment rather than position
                if not result["is_pep"]:
                    result["is_pep"] = True
                    result["pep_category"] = "government_employee"
                    result["confidence"] = 0.5
                    result["risk_level"] = RiskLevel.LOW
                    result["matches"].append(f"Government affiliation: {indicator}")
                break
        
        return result
    
    def _update_state_with_pep_results(
        self, 
        state: AMLState, 
        customer_result: Dict,
        beneficial_owner_results: List,
        party_results: List
    ) -> AMLState:
        """Update state with PEP screening results"""
        updated_state = state.copy()
        
        # Update primary customer PEP status
        if customer_result["is_pep"]:
            updated_state["pep_status"] = True
            updated_state["pep_category"] = customer_result["pep_category"]
            
            # Add risk factors
            risk_factors = [
                "PEP_CUSTOMER",
                f"PEP_CATEGORY_{customer_result['pep_category'].upper()}"
            ]
            updated_state = self.add_risk_factors(updated_state, risk_factors)
        else:
            updated_state["pep_status"] = False
        
        # Handle beneficial owner PEPs
        if beneficial_owner_results:
            risk_factors = []
            for owner, result in beneficial_owner_results:
                risk_factors.extend([
                    "PEP_BENEFICIAL_OWNER",
                    f"PEP_OWNER_CATEGORY_{result['pep_category'].upper()}"
                ])
            
            updated_state = self.add_risk_factors(updated_state, risk_factors)
        
        # Handle transaction party PEPs
        if party_results:
            risk_factors = []
            for party, result in party_results:
                risk_factors.extend([
                    "PEP_TRANSACTION_PARTY",
                    f"PEP_PARTY_CATEGORY_{result['pep_category'].upper()}"
                ])
            
            updated_state = self.add_risk_factors(updated_state, risk_factors)
        
        # Store detailed PEP analysis in metadata
        updated_state["metadata"]["pep_analysis"] = {
            "customer_pep": customer_result,
            "beneficial_owner_peps": beneficial_owner_results,
            "party_peps": party_results
        }
        
        return updated_state
    
    def _generate_pep_alerts(
        self,
        state: AMLState,
        customer_result: Dict,
        beneficial_owner_results: List,
        party_results: List
    ) -> AMLState:
        """Generate appropriate PEP-related alerts"""
        alerts_to_add = []
        transaction = state["transaction"]
        customer = state["customer"]
        
        # Customer PEP alert
        if customer_result["is_pep"]:
            severity = customer_result["risk_level"]
            confidence = customer_result["confidence"]
            
            alert = self.create_alert(
                AlertType.PEP_DETECTED,
                f"Customer identified as PEP: {customer_result['pep_category']} (confidence: {confidence:.1%})",
                severity=severity,
                confidence=confidence,
                evidence=[f"PEP matches: {', '.join(customer_result['matches'])}"],
                transaction_id=transaction.transaction_id,
                customer_id=customer.customer_id
            )
            alerts_to_add.append(alert)
        
        # Beneficial owner PEP alerts
        for owner, result in beneficial_owner_results:
            alert = self.create_alert(
                AlertType.PEP_DETECTED,
                f"Beneficial owner identified as PEP: {owner.owner_name} ({result['pep_category']})",
                severity=result["risk_level"],
                confidence=result["confidence"],
                evidence=[f"Owner PEP matches: {', '.join(result['matches'])}"],
                transaction_id=transaction.transaction_id,
                customer_id=customer.customer_id
            )
            alerts_to_add.append(alert)
        
        # Transaction party PEP alerts  
        for party, result in party_results:
            alert = self.create_alert(
                AlertType.PEP_DETECTED,
                f"Transaction party identified as PEP: {party} ({result['pep_category']})",
                severity=result["risk_level"],
                confidence=result["confidence"],
                evidence=[f"Party PEP matches: {', '.join(result['matches'])}"],
                transaction_id=transaction.transaction_id,
                customer_id=customer.customer_id
            )
            alerts_to_add.append(alert)
        
        # Add all alerts to state
        updated_state = state
        for alert in alerts_to_add:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _assess_edd_requirements(self, state: AMLState) -> AMLState:
        """Assess enhanced due diligence requirements based on PEP findings"""
        pep_analysis = state["metadata"].get("pep_analysis", {})
        
        edd_required = False
        edd_reasons = []
        
        # Customer PEP requires EDD
        if state.get("pep_status"):
            edd_required = True
            edd_reasons.append("Customer is a PEP")
        
        # Beneficial owner PEPs require EDD
        if pep_analysis.get("beneficial_owner_peps"):
            edd_required = True
            edd_reasons.append("Beneficial owner(s) are PEPs")
        
        # High-risk PEP categories require enhanced EDD
        high_risk_categories = ["senior_political_figures", "judicial_officials"]
        
        for analysis in [pep_analysis.get("customer_pep", {})] + [
            result for _, result in pep_analysis.get("beneficial_owner_peps", [])
        ]:
            if analysis.get("pep_category") in high_risk_categories:
                edd_reasons.append(f"High-risk PEP category: {analysis['pep_category']}")
        
        # Update state with EDD assessment
        if edd_required:
            updated_state = state.copy()
            updated_state["metadata"]["edd_required"] = True
            updated_state["metadata"]["edd_reasons"] = edd_reasons
            
            return self.add_risk_factors(updated_state, ["EDD_REQUIRED_PEP"])
        
        return state