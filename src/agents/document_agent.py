"""
Document Analysis Agent for AML Detection System

This agent specializes in analyzing transaction documents to detect
trade-based money laundering, invoice manipulation, and document fraud.
"""
import asyncio
import re
from typing import List, Dict, Any, Tuple

from .base_agent import BaseAgent
from ..core.state_manager import AMLState
from ..models.enums import AlertType, RiskLevel, AnalysisType
from ..services.llm_service import LLMService
from config.settings import DOCUMENT_RISK_PATTERNS


class DocumentAnalysisAgent(BaseAgent):
    """
    Specialized agent for document analysis and trade-based laundering detection.
    
    Responsibilities:
    - Analyze transaction documents for inconsistencies
    - Detect trade-based money laundering indicators
    - Identify document fraud and manipulation
    - Screen for prohibited goods and services
    - Assess invoice and shipment documentation
    """
    
    def __init__(self, llm_service: LLMService):
        """Initialize the document analysis agent"""
        super().__init__(llm_service, "document_analyzer")
        
        # Document risk patterns from configuration
        self.risk_patterns = set(DOCUMENT_RISK_PATTERNS)
        
        # Trade-based laundering indicators
        self.tbml_indicators = {
            "over_invoicing": [
                "overpriced", "inflated price", "above market value",
                "excessive cost", "unrealistic pricing"
            ],
            "under_invoicing": [
                "underpriced", "below market value", "nominal fee",
                "token amount", "significantly discounted"
            ],
            "phantom_shipment": [
                "no shipment", "virtual goods", "intangible services",
                "consulting fee", "management fee", "license fee"
            ],
            "multiple_invoicing": [
                "duplicate invoice", "re-billing", "additional charges",
                "supplemental invoice", "amended invoice"
            ]
        }
        
        # Prohibited goods indicators
        self.prohibited_goods = {
            "weapons": ["weapon", "arms", "ammunition", "explosive", "military"],
            "drugs": ["narcotics", "controlled substance", "pharmaceutical", "precursor"],
            "wildlife": ["ivory", "rhino horn", "endangered species", "wildlife"],
            "cultural": ["artifact", "antiquity", "cultural property", "stolen art"],
            "dual_use": ["dual-use technology", "sensitive technology", "export control"]
        }
        
        # Shell company indicators
        self.shell_indicators = [
            "nominee director", "bearer shares", "minimal activity",
            "brass plate company", "mailbox company", "letterbox company",
            "accommodation address", "virtual office"
        ]
    
    def analyze(self, state: AMLState) -> AMLState:
        """
        Perform comprehensive document analysis.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with document analysis results
        """
        return asyncio.run(self.analyze_documents_async(state))
    
    async def analyze_documents_async(self, state: AMLState) -> AMLState:
        """
        Asynchronously analyze transaction documents for AML risks.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with document analysis results
        """
        if not self.validate_state(state):
            return state
        
        self.log_analysis_start(state, "document_analysis")
        
        # Update path tracking
        state = self.update_path(state, "document_analysis")
        
        try:
            transaction = state["transaction"]
            
            # Check if documents are available
            if not transaction.documents or len(transaction.documents) == 0:
                return self._handle_missing_documents(state)
            
            # Analyze each document
            document_results = []
            for document in transaction.documents:
                result = await self._analyze_single_document(state, document)
                document_results.append(result)
            
            # Consolidate results
            state = self._consolidate_document_analysis(state, document_results)
            
            # Perform cross-document analysis
            state = self._perform_cross_document_analysis(state, document_results)
            
            # Generate comprehensive document risk assessment
            state = await self._generate_document_risk_assessment(state)
            
            self.log_analysis_complete(
                state,
                "document_analysis",
                findings_count=len([rf for rf in state["risk_factors"] if "DOCUMENT" in rf]),
                alerts_generated=len([a for a in state["alerts"] if "document" in a.description.lower()])
            )
            
            return state
            
        except Exception as e:
            self.log_error(state, f"Error in document analysis: {str(e)}", e)
            return state
    
    def _handle_missing_documents(self, state: AMLState) -> AMLState:
        """Handle cases where documents are missing"""
        transaction = state["transaction"]
        
        # Check if documents are required based on transaction characteristics
        requires_docs = (
            transaction.amount > 25000 or  # Large transactions
            transaction.origin_country != transaction.destination_country or  # Cross-border
            len(transaction.intermediate_countries) > 0  # Complex routing
        )
        
        if requires_docs:
            alert = self.create_alert(
                AlertType.MISSING_DOCUMENTS,
                f"Required documentation missing for transaction of {transaction.amount} {transaction.currency}",
                severity=RiskLevel.MEDIUM,
                confidence=0.8,
                evidence=["Large/complex transaction without supporting documentation"],
                transaction_id=transaction.transaction_id,
                customer_id=state["customer"].customer_id
            )
            
            state = self.add_alert_to_state(state, alert)
            state = self.add_risk_factors(state, ["MISSING_REQUIRED_DOCUMENTS"])
        
        return state
    
    async def _analyze_single_document(self, state: AMLState, document) -> Dict[str, Any]:
        """Analyze a single document for risk indicators"""
        analysis_result = {
            "document_id": document.document_id,
            "document_type": document.document_type.value,
            "risk_indicators": [],
            "findings": [],
            "confidence": 0.0
        }
        
        try:
            document_content = document.document_content.lower()
            
            # Pattern-based analysis
            pattern_risks = self._detect_risk_patterns(document_content)
            analysis_result["risk_indicators"].extend(pattern_risks)
            
            # Trade-based laundering detection
            tbml_risks = self._detect_tbml_indicators(document_content)
            analysis_result["risk_indicators"].extend(tbml_risks)
            
            # Prohibited goods screening
            prohibited_risks = self._screen_prohibited_goods(document_content)
            analysis_result["risk_indicators"].extend(prohibited_risks)
            
            # Shell company indicators
            shell_risks = self._detect_shell_indicators(document_content)
            analysis_result["risk_indicators"].extend(shell_risks)
            
            # LLM-powered analysis for complex patterns
            if analysis_result["risk_indicators"]:
                llm_analysis = await self._perform_llm_document_analysis(document_content)
                analysis_result["llm_findings"] = llm_analysis
                analysis_result["confidence"] = 0.8
            
        except Exception as e:
            self.log_error(state, f"Error analyzing document {document.document_id}: {str(e)}", e)
            analysis_result["error"] = str(e)
        
        return analysis_result
    
    def _detect_risk_patterns(self, content: str) -> List[str]:
        """Detect document risk patterns using keyword matching"""
        detected_patterns = []
        
        for pattern in self.risk_patterns:
            pattern_keywords = pattern.lower().replace("_", " ").split()
            
            # Check if all keywords in pattern are present
            if all(keyword in content for keyword in pattern_keywords):
                detected_patterns.append(pattern)
        
        return detected_patterns
    
    def _detect_tbml_indicators(self, content: str) -> List[str]:
        """Detect trade-based money laundering indicators"""
        detected_indicators = []
        
        for category, keywords in self.tbml_indicators.items():
            for keyword in keywords:
                if keyword in content:
                    detected_indicators.append(f"TBML_{category.upper()}")
                    break  # Avoid duplicates for same category
        
        return detected_indicators
    
    def _screen_prohibited_goods(self, content: str) -> List[str]:
        """Screen for prohibited goods and services"""
        detected_goods = []
        
        for category, keywords in self.prohibited_goods.items():
            for keyword in keywords:
                if keyword in content:
                    detected_goods.append(f"PROHIBITED_{category.upper()}")
                    break  # Avoid duplicates for same category
        
        return detected_goods
    
    def _detect_shell_indicators(self, content: str) -> List[str]:
        """Detect shell company indicators"""
        detected_indicators = []
        
        for indicator in self.shell_indicators:
            if indicator in content:
                detected_indicators.append("SHELL_COMPANY_INDICATOR")
        
        return list(set(detected_indicators))  # Remove duplicates
    
    async def _perform_llm_document_analysis(self, content: str) -> str:
        """Perform LLM-powered document analysis for complex patterns"""
        analysis_prompt = f"""
        Analyze this document content for Anti-Money Laundering (AML) risk indicators:
        
        Document Content: {content[:2000]}  # Limit content for token efficiency
        
        Look for:
        1. Invoice manipulation or pricing irregularities
        2. Trade-based money laundering indicators
        3. Shell company or nominee arrangements
        4. Prohibited goods or services
        5. Documentation inconsistencies
        6. Unusual payment terms or structures
        
        Provide specific risk indicators in UPPERCASE format.
        Focus on practical AML concerns that require investigation.
        """
        
        try:
            response = await self.llm_service.analyze_text_async(
                analysis_prompt,
                analysis_type="document_analysis"
            )
            return response
        except Exception as e:
            self.log_error({}, f"Error in LLM document analysis: {str(e)}", e)
            return "LLM analysis unavailable"
    
    def _consolidate_document_analysis(self, state: AMLState, document_results: List[Dict[str, Any]]) -> AMLState:
        """Consolidate analysis results from multiple documents"""
        all_risk_indicators = []
        all_findings = []
        alerts_to_add = []
        
        for result in document_results:
            all_risk_indicators.extend(result.get("risk_indicators", []))
            all_findings.extend(result.get("findings", []))
            
            # Generate alerts for high-risk documents
            if result.get("risk_indicators"):
                severity = RiskLevel.MEDIUM
                
                # Escalate severity for critical indicators
                critical_indicators = [
                    indicator for indicator in result["risk_indicators"]
                    if any(critical in indicator for critical in [
                        "PROHIBITED", "SHELL_COMPANY", "TBML"
                    ])
                ]
                
                if critical_indicators:
                    severity = RiskLevel.HIGH
                
                alert = self.create_alert(
                    AlertType.TRADE_ANOMALY,
                    f"Document {result['document_id']} contains {len(result['risk_indicators'])} risk indicators",
                    severity=severity,
                    confidence=result.get("confidence", 0.7),
                    evidence=result["risk_indicators"],
                    transaction_id=state["transaction"].transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts_to_add.append(alert)
        
        # Update state with consolidated results
        updated_state = self.add_risk_factors(state, list(set(all_risk_indicators)))
        
        for alert in alerts_to_add:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        # Store document analysis results in metadata
        updated_state["document_analysis_results"] = {
            "total_documents": len(document_results),
            "documents_with_risks": len([r for r in document_results if r.get("risk_indicators")]),
            "unique_risk_indicators": list(set(all_risk_indicators)),
            "detailed_results": document_results
        }
        
        return updated_state
    
    def _perform_cross_document_analysis(self, state: AMLState, document_results: List[Dict[str, Any]]) -> AMLState:
        """Perform analysis across multiple documents to detect inconsistencies"""
        if len(document_results) < 2:
            return state
        
        # Look for inconsistencies between documents
        inconsistencies = []
        
        # Check for conflicting information (simplified analysis)
        invoice_docs = [r for r in document_results if "invoice" in r.get("document_type", "").lower()]
        shipping_docs = [r for r in document_results if "shipping" in r.get("document_type", "").lower()]
        
        if invoice_docs and shipping_docs:
            # This is a simplified check - in production, would parse actual values
            inconsistencies.append("POTENTIAL_INVOICE_SHIPPING_MISMATCH")
        
        # Add inconsistencies as risk factors
        if inconsistencies:
            updated_state = self.add_risk_factors(state, inconsistencies)
            
            alert = self.create_alert(
                AlertType.INVOICE_MISMATCH,
                f"Cross-document analysis identified {len(inconsistencies)} potential inconsistencies",
                severity=RiskLevel.MEDIUM,
                confidence=0.7,
                evidence=inconsistencies,
                transaction_id=state["transaction"].transaction_id,
                customer_id=state["customer"].customer_id
            )
            
            return self.add_alert_to_state(updated_state, alert)
        
        return state
    
    async def _generate_document_risk_assessment(self, state: AMLState) -> AMLState:
        """Generate comprehensive document risk assessment using LLM"""
        document_risks = [rf for rf in state["risk_factors"] if any(
            pattern in rf for pattern in ["DOCUMENT", "TBML", "PROHIBITED", "SHELL", "INVOICE"]
        )]
        
        if not document_risks:
            return state
        
        assessment_prompt = f"""
        Provide a comprehensive AML document risk assessment based on the following findings:
        
        Document Risk Indicators: {document_risks}
        Transaction Amount: {state['transaction'].amount} {state['transaction'].currency}
        Cross-border: {state['transaction'].origin_country != state['transaction'].destination_country}
        Number of Documents: {len(state['transaction'].documents)}
        
        Assessment should cover:
        1. Overall document integrity and consistency
        2. Trade-based money laundering risk level
        3. Specific compliance concerns
        4. Recommended investigative actions
        5. Regulatory reporting considerations
        
        Provide practical, actionable insights for AML compliance officers.
        """
        
        try:
            response = await self.llm_service.analyze_text_async(
                assessment_prompt,
                analysis_type="document_risk_assessment"
            )
            
            # Extract additional risk indicators from assessment
            assessment_risks = self.extract_risk_indicators_from_text(response)
            
            # Create comprehensive LLM analysis result
            llm_analysis = self.create_llm_analysis_result(
                analysis_type=AnalysisType.DOCUMENT_ANALYSIS,
                analysis_text=response,
                key_findings=document_risks,
                risk_indicators=assessment_risks,
                confidence_score=0.85,
                model_used=self.llm_service.get_model_name()
            )
            
            # Add to state
            state = self.add_llm_analysis_to_state(state, llm_analysis)
            
        except Exception as e:
            self.log_error(state, f"Error in document risk assessment: {str(e)}", e)
        
        return state