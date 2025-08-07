"""
Workflow Engine for AML Detection System

This module implements the LangGraph-based workflow engine that orchestrates
the AI Squadron AML detection process using coordinated multi-agent principles.
"""
import asyncio
from datetime import datetime
from typing import Dict, Any, Callable, Optional

from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI

from ..core.state_manager import AMLState, StateManager
from ..core.risk_calculator import RiskCalculator
from ..agents.orchestrator_agent import OrchestratorAgent
from ..agents.sanctions_agent import SanctionsScreeningAgent
from ..agents.pep_agent import PEPScreeningAgent
from ..agents.geographic_agent import GeographicRiskAgent
from ..agents.behavioral_agent import BehavioralAnalysisAgent
from ..agents.crypto_agent import CryptoRiskAgent
from ..agents.document_agent import DocumentAnalysisAgent
from ..agents.enhanced_dd_agent import EnhancedDueDiligenceAgent
from ..services.llm_service import LLMService
from config.settings import settings


class WorkflowEngine:
    """
    Orchestrates the AML detection workflow using LangGraph and specialized agents.
    Implements the seven principles of ambient intelligence through AI Squadron coordination.
    """
    
    def __init__(self):
        """Initialize the workflow engine with all necessary components"""
        self.state_manager = StateManager()
        self.risk_calculator = RiskCalculator()
        self.llm_service = LLMService()
        
        # Initialize specialized agents
        self._initialize_agents()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _initialize_agents(self):
        """Initialize all specialized AML agents"""
        self.orchestrator = OrchestratorAgent(self.llm_service)
        self.sanctions_agent = SanctionsScreeningAgent(self.llm_service)
        self.pep_agent = PEPScreeningAgent(self.llm_service)
        self.geographic_agent = GeographicRiskAgent(self.llm_service)
        self.behavioral_agent = BehavioralAnalysisAgent(self.llm_service)
        self.crypto_agent = CryptoRiskAgent(self.llm_service)
        self.document_agent = DocumentAnalysisAgent(self.llm_service)
        self.edd_agent = EnhancedDueDiligenceAgent(self.llm_service)
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow for AML detection.
        Implements conditional routing based on transaction characteristics.
        """
        workflow = StateGraph(AMLState)
        
        # Register all workflow nodes
        nodes = [
            ("initial_screening", self._initial_screening),
            ("crypto_analysis", self._crypto_analysis),
            ("geo_analysis", self._geographic_analysis),
            ("document_check", self._document_analysis),
            ("behavior_check", self._behavioral_analysis),
            ("sanctions_check", self._sanctions_screening),
            ("pep_check", self._pep_screening),
            ("enhanced_dd", self._enhanced_due_diligence),
            ("risk_scoring", self._risk_scoring),
            ("generate_sar", self._generate_sar),
            ("human_review", self._human_review),
            ("approve_transaction", self._approve_transaction)
        ]
        
        for name, func in nodes:
            workflow.add_node(name, func)
        
        # Set entry point
        workflow.set_entry_point("initial_screening")
        
        # Add conditional edges for workflow routing
        self._add_conditional_edges(workflow)
        
        # Add standard edges
        self._add_standard_edges(workflow)
        
        return workflow
    
    def _add_conditional_edges(self, workflow: StateGraph):
        """Add conditional routing edges to the workflow"""
        
        # Initial screening routing
        workflow.add_conditional_edges(
            "initial_screening",
            self._route_initial_screening,
            {
                "CRYPTO_TRANSACTION": "crypto_analysis",
                "LARGE_TRANSACTION": "geo_analysis", 
                "NEW_ACCOUNT": "enhanced_dd",
                "STANDARD_FLOW": "document_check"
            }
        )
        
        # Crypto analysis routing
        workflow.add_conditional_edges(
            "crypto_analysis",
            self._route_crypto_analysis,
            {
                "HIGH_RISK_CRYPTO": "enhanced_dd",
                "NORMAL_CRYPTO": "document_check"
            }
        )
        
        # Sanctions check routing
        workflow.add_conditional_edges(
            "sanctions_check",
            self._route_sanctions_check,
            {
                "SANCTION_HIT": "generate_sar",
                "NO_HIT": "pep_check"
            }
        )
        
        # PEP check routing
        workflow.add_conditional_edges(
            "pep_check",
            self._route_pep_check,
            {
                "PEP_FOUND": "enhanced_dd",
                "NO_PEP": "risk_scoring"
            }
        )
        
        # Risk scoring routing (final decision)
        workflow.add_conditional_edges(
            "risk_scoring",
            self._route_final_decision,
            {
                "HIGH_RISK": "generate_sar",
                "MEDIUM_RISK": "human_review",
                "LOW_RISK": "approve_transaction"
            }
        )
    
    def _add_standard_edges(self, workflow: StateGraph):
        """Add standard (non-conditional) edges to the workflow"""
        workflow.add_edge("geo_analysis", "document_check")
        workflow.add_edge("document_check", "behavior_check")
        workflow.add_edge("behavior_check", "sanctions_check")
        workflow.add_edge("enhanced_dd", "risk_scoring")
        workflow.add_edge("generate_sar", END)
        workflow.add_edge("human_review", END)
        workflow.add_edge("approve_transaction", END)
    
    # Node Implementation Functions
    
    def _initial_screening(self, state: AMLState) -> AMLState:
        """Initial screening and workflow entry point"""
        state = self.state_manager.update_path(state, "initial_screening")
        return self.orchestrator.initial_assessment(state)
    
    def _crypto_analysis(self, state: AMLState) -> AMLState:
        """Cryptocurrency-specific risk analysis"""
        state = self.state_manager.update_path(state, "crypto_analysis")
        return self.crypto_agent.analyze_crypto_risks(state)
    
    def _geographic_analysis(self, state: AMLState) -> AMLState:
        """Geographic and jurisdictional risk analysis"""
        state = self.state_manager.update_path(state, "geographic_analysis")
        return self.geographic_agent.assess_geographic_risks(state)
    
    def _document_analysis(self, state: AMLState) -> AMLState:
        """Document analysis for trade-based laundering detection"""
        state = self.state_manager.update_path(state, "document_analysis")
        return asyncio.run(self.document_agent.analyze_documents_async(state))
    
    def _behavioral_analysis(self, state: AMLState) -> AMLState:
        """Behavioral pattern analysis for structuring detection"""
        state = self.state_manager.update_path(state, "behavioral_analysis")
        return self.behavioral_agent.analyze_patterns(state)
    
    def _sanctions_screening(self, state: AMLState) -> AMLState:
        """Sanctions list screening"""
        state = self.state_manager.update_path(state, "sanctions_screening")
        return self.sanctions_agent.screen_sanctions(state)
    
    def _pep_screening(self, state: AMLState) -> AMLState:
        """Politically Exposed Person screening"""
        state = self.state_manager.update_path(state, "pep_screening")
        return self.pep_agent.screen_pep(state)
    
    def _enhanced_due_diligence(self, state: AMLState) -> AMLState:
        """Enhanced due diligence investigation"""
        state = self.state_manager.update_path(state, "enhanced_due_diligence")
        return asyncio.run(self.edd_agent.conduct_edd_async(state))
    
    def _risk_scoring(self, state: AMLState) -> AMLState:
        """Comprehensive risk scoring using weighted factors"""
        state = self.state_manager.update_path(state, "risk_scoring")
        
        # Calculate comprehensive risk assessment
        risk_assessment = self.risk_calculator.calculate_comprehensive_risk(state)
        
        # Update state with risk assessment
        updated_state = state.copy()
        updated_state["risk_assessment"] = risk_assessment
        updated_state["risk_score"] = risk_assessment.risk_score
        updated_state["risk_level"] = risk_assessment.risk_level
        
        return updated_state
    
    def _generate_sar(self, state: AMLState) -> AMLState:
        """Generate Suspicious Activity Report"""
        state = self.state_manager.update_path(state, "sar_generation")
        
        # Generate case ID and set SAR status
        case_id = self._generate_case_id()
        updated_state = state.copy()
        updated_state["case_id"] = case_id
        updated_state["reporting_status"] = "SAR_FILED"
        updated_state["sar_timestamp"] = datetime.utcnow()
        
        # Set final status
        return self.state_manager.set_final_status(
            updated_state,
            status="SAR_GENERATED",
            sar_recommended=True,
            approved=False
        )
    
    def _human_review(self, state: AMLState) -> AMLState:
        """Queue for human review"""
        state = self.state_manager.update_path(state, "human_review")
        
        # Set review deadline (24 hours from now)
        review_deadline = datetime.utcnow()
        updated_state = state.copy()
        updated_state["review_status"] = "PENDING"
        updated_state["review_deadline"] = review_deadline
        
        # Set final status
        return self.state_manager.set_final_status(
            updated_state,
            status="PENDING_REVIEW",
            requires_review=True,
            approved=False
        )
    
    def _approve_transaction(self, state: AMLState) -> AMLState:
        """Approve transaction for processing"""
        state = self.state_manager.update_path(state, "transaction_approval")
        
        return self.state_manager.set_final_status(
            state,
            status="APPROVED",
            approved=True
        )
    
    # Routing Functions
    
    def _route_initial_screening(self, state: AMLState) -> str:
        """Route based on initial transaction screening"""
        transaction = state["transaction"]
        customer = state["customer"]
        
        if transaction.asset_type.value == "CRYPTO":
            return "CRYPTO_TRANSACTION"
        elif transaction.amount > 100000:  # Large transaction threshold
            return "LARGE_TRANSACTION"
        elif customer.account_age_days < 7:  # New account
            return "NEW_ACCOUNT"
        else:
            return "STANDARD_FLOW"
    
    def _route_crypto_analysis(self, state: AMLState) -> str:
        """Route based on crypto analysis results"""
        crypto_risk_indicators = [
            rf for rf in state["risk_factors"] 
            if "CRYPTO" in rf or "MIXER" in rf or "DARKNET" in rf
        ]
        
        if crypto_risk_indicators:
            return "HIGH_RISK_CRYPTO"
        else:
            return "NORMAL_CRYPTO"
    
    def _route_sanctions_check(self, state: AMLState) -> str:
        """Route based on sanctions screening results"""
        if state["sanction_hits"]:
            return "SANCTION_HIT"
        else:
            return "NO_HIT"
    
    def _route_pep_check(self, state: AMLState) -> str:
        """Route based on PEP screening results"""
        if state["pep_status"]:
            return "PEP_FOUND"
        else:
            return "NO_PEP"
    
    def _route_final_decision(self, state: AMLState) -> str:
        """Route based on final risk score"""
        risk_score = state["risk_score"]
        
        if risk_score >= settings.risk.high_risk_threshold:
            return "HIGH_RISK"
        elif risk_score >= settings.risk.medium_risk_threshold:
            return "MEDIUM_RISK"
        else:
            return "LOW_RISK"
    
    def _generate_case_id(self) -> str:
        """Generate unique case ID for SAR or investigation"""
        import hashlib
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:12]
    
    # Public Interface
    
    async def analyze_transaction_async(self, state: AMLState) -> AMLState:
        """
        Asynchronously execute the AML analysis workflow.
        
        Args:
            state: Initial AML state with transaction and customer data
            
        Returns:
            Final state after completing workflow
        """
        try:
            # Execute the compiled workflow
            result = await self.app.ainvoke(state)
            return result
        except Exception as e:
            # Handle workflow execution errors
            error_state = state.copy()
            error_state["final_status"] = "ERROR"
            error_state["error_message"] = str(e)
            error_state["requires_human_review"] = True
            return error_state
    
    def analyze_transaction(self, state: AMLState) -> AMLState:
        """
        Synchronously execute the AML analysis workflow.
        
        Args:
            state: Initial AML state with transaction and customer data
            
        Returns:
            Final state after completing workflow
        """
        try:
            # Execute the compiled workflow
            result = self.app.invoke(state)
            return result
        except Exception as e:
            # Handle workflow execution errors
            error_state = state.copy()
            error_state["final_status"] = "ERROR"
            error_state["error_message"] = str(e)
            error_state["requires_human_review"] = True
            return error_state
    
    def get_workflow_visualization(self) -> str:
        """
        Get a text representation of the workflow for debugging.
        
        Returns:
            String representation of the workflow graph
        """
        # This would return a visualization of the workflow
        # For now, return a simple description
        return """
        AML Detection Workflow:
        
        initial_screening -> [crypto_analysis | geo_analysis | enhanced_dd | document_check]
        crypto_analysis -> [enhanced_dd | document_check]
        geo_analysis -> document_check
        document_check -> behavior_check
        behavior_check -> sanctions_check
        sanctions_check -> [generate_sar | pep_check]
        pep_check -> [enhanced_dd | risk_scoring]
        enhanced_dd -> risk_scoring
        risk_scoring -> [generate_sar | human_review | approve_transaction]
        """