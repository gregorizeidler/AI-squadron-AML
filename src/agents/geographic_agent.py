"""
Geographic Risk Analysis Agent for AML Detection System

This agent specializes in analyzing geographic and jurisdictional risks
associated with transaction routing, country risk assessments, and sanctions.
"""
from typing import List, Dict, Set
from decimal import Decimal

from .base_agent import BaseAgent
from ..core.state_manager import AMLState
from ..models.enums import AlertType, RiskLevel, AnalysisType
from ..services.llm_service import LLMService
from config.settings import HIGH_RISK_COUNTRIES, TAX_HAVENS


class GeographicRiskAgent(BaseAgent):
    """
    Specialized agent for geographic and jurisdictional risk analysis.
    
    Responsibilities:
    - Assess country-specific money laundering risks
    - Analyze transaction routing patterns
    - Identify tax haven involvement
    - Screen for sanctioned jurisdictions
    - Evaluate cross-border transaction risks
    - Detect unusual geographic patterns
    """
    
    def __init__(self, llm_service: LLMService):
        """Initialize the geographic risk agent"""
        super().__init__(llm_service, "geographic_analyst")
        
        # Extended high-risk countries
        self.high_risk_countries = set(HIGH_RISK_COUNTRIES + [
            "AF", "BD", "BO", "GH", "GY", "LK", "MV", "NP", "PK", "LB", "UG"
        ])
        
        # Extended tax havens list
        self.tax_havens = set(TAX_HAVENS + [
            "AD", "AG", "AI", "AS", "BB", "BH", "BZ", "CK", "CW", "GI",
            "HK", "IM", "IO", "KN", "LC", "LI", "MC", "MS", "NR", "NU",
            "PW", "SC", "SG", "SM", "TC", "TO", "TT", "TV", "UM", "VC", "VU", "WS"
        ])
        
        # Sanctioned and embargoed countries
        self.sanctioned_countries = {
            "IR", "KP", "SY", "CU", "RU", "BY"  # Iran, North Korea, Syria, Cuba, Russia, Belarus
        }
        
        # Countries with weak AML controls
        self.weak_aml_countries = {
            "MM", "AF", "KH", "LA", "TD", "GN", "BI", "SO", "SS", "YE"
        }
        
        # Common money laundering routes
        self.known_laundering_routes = [
            ["CN", "HK", "VG"],  # China -> Hong Kong -> British Virgin Islands
            ["RU", "CY", "MT"],  # Russia -> Cyprus -> Malta  
            ["BR", "PA", "KY"],  # Brazil -> Panama -> Cayman Islands
            ["NG", "AE", "CH"],  # Nigeria -> UAE -> Switzerland
            ["MX", "US", "KY"],  # Mexico -> US -> Cayman Islands
        ]
    
    def analyze(self, state: AMLState) -> AMLState:
        """
        Perform comprehensive geographic risk analysis.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with geographic analysis results
        """
        return self.assess_geographic_risks(state)
    
    def assess_geographic_risks(self, state: AMLState) -> AMLState:
        """
        Assess geographic and jurisdictional risks for the transaction.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with geographic risk assessment
        """
        if not self.validate_state(state):
            return state
        
        self.log_analysis_start(state, "geographic_risk_assessment")
        
        # Update path tracking
        state = self.update_path(state, "geographic_analysis")
        
        try:
            transaction = state["transaction"]
            
            # Collect all jurisdictions involved
            all_jurisdictions = self._collect_all_jurisdictions(transaction)
            
            # Assess individual country risks
            state = self._assess_individual_country_risks(state, all_jurisdictions)
            
            # Analyze transaction routing patterns
            state = self._analyze_routing_patterns(state)
            
            # Check for known laundering routes
            state = self._check_known_laundering_routes(state)
            
            # Assess sanctions compliance
            state = self._assess_sanctions_compliance(state, all_jurisdictions)
            
            # Evaluate cross-border complexity
            state = self._evaluate_cross_border_complexity(state)
            
            # Generate geographic risk assessment
            state = self._generate_geographic_risk_assessment(state, all_jurisdictions)
            
            self.log_analysis_complete(
                state,
                "geographic_risk_assessment",
                findings_count=len([rf for rf in state["risk_factors"] if "GEO" in rf or "COUNTRY" in rf]),
                alerts_generated=len([a for a in state["alerts"] if "geographic" in a.description.lower()])
            )
            
            return state
            
        except Exception as e:
            self.log_error(state, f"Error in geographic risk assessment: {str(e)}", e)
            return state
    
    def _collect_all_jurisdictions(self, transaction) -> List[str]:
        """Collect all jurisdictions involved in the transaction"""
        jurisdictions = []
        
        # Add origin and destination
        jurisdictions.append(transaction.origin_country)
        jurisdictions.append(transaction.destination_country)
        
        # Add intermediate countries
        jurisdictions.extend(transaction.intermediate_countries)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_jurisdictions = []
        for jurisdiction in jurisdictions:
            if jurisdiction not in seen:
                unique_jurisdictions.append(jurisdiction)
                seen.add(jurisdiction)
        
        return unique_jurisdictions
    
    def _assess_individual_country_risks(self, state: AMLState, jurisdictions: List[str]) -> AMLState:
        """Assess risks for each individual jurisdiction"""
        risks = []
        alerts = []
        transaction = state["transaction"]
        customer = state["customer"]
        
        for jurisdiction in jurisdictions:
            country_risks = []
            
            # High-risk country assessment
            if jurisdiction in self.high_risk_countries:
                country_risks.append(f"HIGH_RISK_COUNTRY_{jurisdiction}")
                
                alert = self.create_alert(
                    AlertType.HIGH_RISK_COUNTRY,
                    f"Transaction involves high-risk jurisdiction: {jurisdiction}",
                    severity=RiskLevel.HIGH,
                    confidence=0.9,
                    evidence=[f"Country {jurisdiction} classified as high-risk for money laundering"],
                    transaction_id=transaction.transaction_id,
                    customer_id=customer.customer_id
                )
                alerts.append(alert)
            
            # Tax haven assessment
            if jurisdiction in self.tax_havens:
                country_risks.append(f"TAX_HAVEN_{jurisdiction}")
                
                # Higher severity for large amounts through tax havens
                severity = RiskLevel.HIGH if transaction.amount > Decimal("50000") else RiskLevel.MEDIUM
                
                alert = self.create_alert(
                    AlertType.HIGH_RISK_COUNTRY,
                    f"Transaction involves tax haven jurisdiction: {jurisdiction}",
                    severity=severity,
                    confidence=0.8,
                    evidence=[f"Country {jurisdiction} identified as tax haven"],
                    transaction_id=transaction.transaction_id,
                    customer_id=customer.customer_id
                )
                alerts.append(alert)
            
            # Sanctioned country assessment
            if jurisdiction in self.sanctioned_countries:
                country_risks.append(f"SANCTIONED_COUNTRY_{jurisdiction}")
                
                alert = self.create_alert(
                    AlertType.SANCTIONS_HIT,
                    f"Transaction involves sanctioned jurisdiction: {jurisdiction}",
                    severity=RiskLevel.CRITICAL,
                    confidence=0.95,
                    evidence=[f"Country {jurisdiction} is under sanctions"],
                    transaction_id=transaction.transaction_id,
                    customer_id=customer.customer_id
                )
                alerts.append(alert)
            
            # Weak AML controls assessment
            if jurisdiction in self.weak_aml_countries:
                country_risks.append(f"WEAK_AML_CONTROLS_{jurisdiction}")
                
                alert = self.create_alert(
                    AlertType.HIGH_RISK_COUNTRY,
                    f"Transaction involves jurisdiction with weak AML controls: {jurisdiction}",
                    severity=RiskLevel.MEDIUM,
                    confidence=0.7,
                    evidence=[f"Country {jurisdiction} has insufficient AML/CFT controls"],
                    transaction_id=transaction.transaction_id,
                    customer_id=customer.customer_id
                )
                alerts.append(alert)
            
            risks.extend(country_risks)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _analyze_routing_patterns(self, state: AMLState) -> AMLState:
        """Analyze transaction routing patterns for suspicious characteristics"""
        transaction = state["transaction"]
        risks = []
        alerts = []
        
        # Complex routing analysis
        total_hops = len(transaction.intermediate_countries)
        
        if total_hops >= 3:
            risks.append("COMPLEX_ROUTING_PATTERN")
            
            alert = self.create_alert(
                AlertType.UNUSUAL_PATTERN,
                f"Complex transaction routing through {total_hops} intermediate jurisdictions",
                severity=RiskLevel.HIGH,
                confidence=0.8,
                evidence=[f"Routing: {transaction.origin_country} -> {' -> '.join(transaction.intermediate_countries)} -> {transaction.destination_country}"],
                transaction_id=transaction.transaction_id,
                customer_id=state["customer"].customer_id
            )
            alerts.append(alert)
            
        elif total_hops >= 1:
            risks.append("INTERMEDIATE_ROUTING")
        
        # Unnecessary routing analysis
        if self._is_unnecessary_routing(transaction):
            risks.append("UNNECESSARY_ROUTING")
            
            alert = self.create_alert(
                AlertType.UNUSUAL_PATTERN,
                "Transaction routing appears unnecessarily complex",
                severity=RiskLevel.MEDIUM,
                confidence=0.7,
                evidence=["Routing complexity not justified by business purpose"],
                transaction_id=transaction.transaction_id,
                customer_id=state["customer"].customer_id
            )
            alerts.append(alert)
        
        # Round-trip detection
        if self._detect_round_trip_pattern(transaction):
            risks.append("ROUND_TRIP_PATTERN")
            
            alert = self.create_alert(
                AlertType.UNUSUAL_PATTERN,
                "Potential round-trip transaction pattern detected",
                severity=RiskLevel.HIGH,
                confidence=0.9,
                evidence=["Transaction returns to origin jurisdiction"],
                transaction_id=transaction.transaction_id,
                customer_id=state["customer"].customer_id
            )
            alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _check_known_laundering_routes(self, state: AMLState) -> AMLState:
        """Check if transaction follows known money laundering routes"""
        transaction = state["transaction"]
        
        # Build full route
        full_route = [transaction.origin_country] + transaction.intermediate_countries + [transaction.destination_country]
        
        # Check against known routes
        for known_route in self.known_laundering_routes:
            if self._route_matches_pattern(full_route, known_route):
                risk_factor = f"KNOWN_LAUNDERING_ROUTE_{'-'.join(known_route)}"
                state = self.add_risk_factors(state, [risk_factor])
                
                alert = self.create_alert(
                    AlertType.UNUSUAL_PATTERN,
                    f"Transaction follows known money laundering route: {' -> '.join(known_route)}",
                    severity=RiskLevel.HIGH,
                    confidence=0.85,
                    evidence=[f"Route matches known pattern: {' -> '.join(known_route)}"],
                    transaction_id=transaction.transaction_id,
                    customer_id=state["customer"].customer_id
                )
                
                state = self.add_alert_to_state(state, alert)
                break
        
        return state
    
    def _assess_sanctions_compliance(self, state: AMLState, jurisdictions: List[str]) -> AMLState:
        """Assess sanctions compliance across all jurisdictions"""
        risks = []
        
        # Check for any sanctioned jurisdictions
        sanctioned_found = [j for j in jurisdictions if j in self.sanctioned_countries]
        
        if sanctioned_found:
            risks.extend([f"SANCTIONS_VIOLATION_{country}" for country in sanctioned_found])
            
            # This would typically trigger immediate blocking
            state["metadata"]["sanctions_violation"] = True
            state["metadata"]["sanctioned_jurisdictions"] = sanctioned_found
        
        # Check for sanctions evasion patterns
        if len(jurisdictions) > 2 and any(j in self.sanctioned_countries for j in jurisdictions):
            risks.append("POTENTIAL_SANCTIONS_EVASION")
        
        return self.add_risk_factors(state, risks)
    
    def _evaluate_cross_border_complexity(self, state: AMLState) -> AMLState:
        """Evaluate the complexity of cross-border transactions"""
        transaction = state["transaction"]
        
        if transaction.origin_country == transaction.destination_country:
            return state  # Domestic transaction
        
        risks = []
        
        # Simple cross-border
        if not transaction.intermediate_countries:
            risks.append("CROSS_BORDER_TRANSACTION")
        
        # Complex cross-border with multiple jurisdictions
        total_jurisdictions = len(set([transaction.origin_country] + 
                                    transaction.intermediate_countries + 
                                    [transaction.destination_country]))
        
        if total_jurisdictions >= 4:
            risks.append("HIGHLY_COMPLEX_CROSS_BORDER")
        elif total_jurisdictions >= 3:
            risks.append("COMPLEX_CROSS_BORDER")
        
        # Multiple regulatory regimes
        if self._involves_multiple_regulatory_regimes(transaction):
            risks.append("MULTIPLE_REGULATORY_REGIMES")
        
        return self.add_risk_factors(state, risks)
    
    def _generate_geographic_risk_assessment(self, state: AMLState, jurisdictions: List[str]) -> AMLState:
        """Generate comprehensive geographic risk assessment using LLM"""
        geographic_risks = [rf for rf in state["risk_factors"] if any(
            pattern in rf for pattern in ["GEO", "COUNTRY", "ROUTING", "CROSS_BORDER", "TAX_HAVEN"]
        )]
        
        if not geographic_risks:
            return state
        
        assessment_prompt = f"""
        Provide a comprehensive geographic risk assessment for this AML analysis:
        
        Transaction Route: {state['transaction'].origin_country} -> {' -> '.join(state['transaction'].intermediate_countries)} -> {state['transaction'].destination_country}
        Transaction Amount: {state['transaction'].amount} {state['transaction'].currency}
        Jurisdictions Involved: {jurisdictions}
        Geographic Risk Factors: {geographic_risks}
        
        Assessment should cover:
        1. Overall geographic risk level
        2. Specific jurisdictional concerns
        3. Routing pattern analysis
        4. Sanctions and compliance implications
        5. Enhanced monitoring recommendations
        
        Focus on practical AML implications and regulatory compliance requirements.
        """
        
        try:
            response = self.llm_service.analyze_text(
                assessment_prompt,
                analysis_type="geographic_risk_assessment"
            )
            
            # Extract additional risk indicators
            assessment_risks = self.extract_risk_indicators_from_text(response)
            
            # Create LLM analysis result
            llm_analysis = self.create_llm_analysis_result(
                analysis_type=AnalysisType.GEOGRAPHIC_RISK,
                analysis_text=response,
                key_findings=geographic_risks,
                risk_indicators=assessment_risks,
                confidence_score=0.85,
                model_used=self.llm_service.get_model_name()
            )
            
            # Add to state
            state = self.add_llm_analysis_to_state(state, llm_analysis)
            
        except Exception as e:
            self.log_error(state, f"Error in geographic risk assessment: {str(e)}", e)
        
        return state
    
    def _is_unnecessary_routing(self, transaction) -> bool:
        """Determine if routing appears unnecessarily complex"""
        # Simple heuristic: if there are intermediate countries that don't add obvious value
        if not transaction.intermediate_countries:
            return False
        
        # Check if intermediate countries are geographically logical
        # This is a simplified check - in production would use more sophisticated logic
        return len(transaction.intermediate_countries) > 1
    
    def _detect_round_trip_pattern(self, transaction) -> bool:
        """Detect if transaction appears to be a round-trip"""
        all_countries = [transaction.origin_country] + transaction.intermediate_countries + [transaction.destination_country]
        
        # Check if origin appears later in the route
        return transaction.origin_country in transaction.intermediate_countries
    
    def _route_matches_pattern(self, full_route: List[str], known_route: List[str]) -> bool:
        """Check if transaction route matches a known pattern"""
        if len(full_route) < len(known_route):
            return False
        
        # Check for subsequence match
        for i in range(len(full_route) - len(known_route) + 1):
            if full_route[i:i+len(known_route)] == known_route:
                return True
        
        return False
    
    def _involves_multiple_regulatory_regimes(self, transaction) -> bool:
        """Check if transaction involves multiple different regulatory regimes"""
        # Simplified classification of regulatory regimes
        regimes = {
            "US": ["US"],
            "EU": ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "IE", "PT", "FI", "GR", "LU", "SI", "SK", "EE", "LV", "LT", "CY", "MT"],
            "UK": ["GB"],
            "ASIA_PACIFIC": ["AU", "NZ", "SG", "HK", "JP", "KR"],
            "MIDDLE_EAST": ["AE", "SA", "QA", "KW", "BH", "OM"],
            "OTHER": []  # Catch-all for other countries
        }
        
        all_countries = [transaction.origin_country] + transaction.intermediate_countries + [transaction.destination_country]
        
        # Determine which regimes are involved
        involved_regimes = set()
        for country in all_countries:
            for regime, countries in regimes.items():
                if country in countries:
                    involved_regimes.add(regime)
                    break
            else:
                involved_regimes.add("OTHER")
        
        return len(involved_regimes) > 2