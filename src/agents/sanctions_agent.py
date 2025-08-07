"""
Sanctions Screening Agent for AML Detection System

This agent performs comprehensive sanctions list screening against
multiple global sanctions databases and watchlists.
"""
from typing import List, Set
import re

from .base_agent import BaseAgent
from ..core.state_manager import AMLState
from ..models.enums import AlertType, RiskLevel, AnalysisType
from ..services.llm_service import LLMService
from config.settings import SANCTIONED_ENTITIES


class SanctionsScreeningAgent(BaseAgent):
    """
    Specialized agent for sanctions and watchlist screening.
    
    Responsibilities:
    - Screen transaction parties against sanctions lists
    - Check beneficial owners and related entities
    - Perform fuzzy matching for name variations
    - Generate sanctions-related alerts and risk factors
    """
    
    def __init__(self, llm_service: LLMService):
        """Initialize the sanctions screening agent"""
        super().__init__(llm_service, "sanctions_screener")
        
        # Extended sanctions list for comprehensive screening
        self.sanctions_entities = set(SANCTIONED_ENTITIES + [
            "prohibited_organization",
            "blacklisted_entity", 
            "terror_finance_group",
            "drug_cartel_entity",
            "weapons_trafficker",
            "human_trafficking_org",
            "corruption_network",
            "cybercrime_group",
            "sanctioned_government_official",
            "designated_terrorist_org"
        ])
        
        # High-risk keywords that may indicate sanctions concerns
        self.risk_keywords = {
            "terror", "terrorist", "terrorism", "narcotics", "cartel",
            "weapons", "arms", "trafficking", "sanctions", "embargo",
            "prohibited", "blacklist", "designated", "blocked",
            "frozen", "restricted", "banned", "illicit"
        }
    
    def analyze(self, state: AMLState) -> AMLState:
        """
        Perform comprehensive sanctions screening analysis.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with sanctions screening results
        """
        return self.screen_sanctions(state)
    
    def screen_sanctions(self, state: AMLState) -> AMLState:
        """
        Perform sanctions screening against multiple databases.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with sanctions screening results
        """
        if not self.validate_state(state):
            return state
        
        self.log_analysis_start(state, "sanctions_screening")
        
        # Update path tracking
        state = self.update_path(state, "sanctions_screening")
        
        try:
            transaction = state["transaction"]
            customer = state["customer"]
            
            # Collect all entities to screen
            entities_to_screen = self._collect_screening_entities(state)
            
            # Perform direct name matching
            direct_hits = self._perform_direct_matching(entities_to_screen)
            
            # Perform fuzzy/partial matching
            fuzzy_hits = self._perform_fuzzy_matching(entities_to_screen)
            
            # Screen beneficial owners
            beneficial_owner_hits = self._screen_beneficial_owners(state)
            
            # Combine all hits
            all_hits = direct_hits + fuzzy_hits + beneficial_owner_hits
            
            # Update state with screening results
            state = self._update_state_with_hits(state, all_hits)
            
            # Generate sanctions-related alerts
            state = self._generate_sanctions_alerts(state, all_hits)
            
            # Update sanctions check flag
            updated_state = state.copy()
            updated_state["sanctions_checked"] = True
            
            self.log_analysis_complete(
                state,
                "sanctions_screening",
                findings_count=len(all_hits),
                alerts_generated=1 if all_hits else 0
            )
            
            return updated_state
            
        except Exception as e:
            self.log_error(state, f"Error in sanctions screening: {str(e)}", e)
            return state
    
    def _collect_screening_entities(self, state: AMLState) -> List[str]:
        """Collect all entities that need sanctions screening"""
        entities = []
        
        transaction = state["transaction"]
        customer = state["customer"]
        
        # Add customer name
        entities.append(customer.name)
        
        # Add transaction parties
        entities.extend(transaction.parties)
        
        # Add sender and receiver if not already included
        if transaction.sender_id not in entities:
            entities.append(transaction.sender_id)
        if transaction.receiver_id not in entities:
            entities.append(transaction.receiver_id)
        
        # Add beneficial owners
        for owner in customer.beneficial_owners:
            entities.append(owner.owner_name)
        
        # Clean and deduplicate
        cleaned_entities = []
        for entity in entities:
            if entity and isinstance(entity, str):
                cleaned = entity.strip().lower()
                if cleaned and cleaned not in cleaned_entities:
                    cleaned_entities.append(cleaned)
        
        return cleaned_entities
    
    def _perform_direct_matching(self, entities: List[str]) -> List[str]:
        """Perform direct/exact matching against sanctions lists"""
        hits = []
        
        for entity in entities:
            entity_lower = entity.lower()
            
            # Check against known sanctions entities
            for sanctioned_entity in self.sanctions_entities:
                if sanctioned_entity.lower() in entity_lower:
                    hits.append(f"DIRECT_MATCH:{entity}:{sanctioned_entity}")
        
        return hits
    
    def _perform_fuzzy_matching(self, entities: List[str]) -> List[str]:
        """Perform fuzzy/partial matching for name variations"""
        hits = []
        
        for entity in entities:
            entity_lower = entity.lower()
            
            # Check for high-risk keywords
            for keyword in self.risk_keywords:
                if keyword in entity_lower:
                    hits.append(f"KEYWORD_MATCH:{entity}:{keyword}")
            
            # Check for common variations and obfuscations
            suspicious_patterns = [
                r'\b(al|abu|ibn|bin)[\s\-_]',  # Common name prefixes
                r'\b(ltd|llc|inc|corp|co|sa|gmbh)\b',  # Corporate entities
                r'\b(foundation|charity|fund|trust)\b',  # Organizations
                r'\b(trading|import|export|shipping)\b'  # Business types
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, entity_lower):
                    # Additional screening for entities matching patterns
                    if self._requires_enhanced_screening(entity_lower):
                        hits.append(f"PATTERN_MATCH:{entity}:{pattern}")
        
        return hits
    
    def _screen_beneficial_owners(self, state: AMLState) -> List[str]:
        """Screen beneficial owners against sanctions lists"""
        hits = []
        customer = state["customer"]
        
        for owner in customer.beneficial_owners:
            # Check if owner is already marked as sanctioned
            if owner.is_sanctioned:
                hits.append(f"BENEFICIAL_OWNER_SANCTIONED:{owner.owner_name}")
            
            # Screen owner name
            owner_name_lower = owner.owner_name.lower()
            for sanctioned_entity in self.sanctions_entities:
                if sanctioned_entity.lower() in owner_name_lower:
                    hits.append(f"BENEFICIAL_OWNER_MATCH:{owner.owner_name}:{sanctioned_entity}")
        
        return hits
    
    def _requires_enhanced_screening(self, entity_name: str) -> bool:
        """Determine if entity requires enhanced sanctions screening"""
        # Check for multiple risk keywords
        keyword_count = sum(1 for keyword in self.risk_keywords if keyword in entity_name)
        
        # Enhanced screening if multiple keywords or specific high-risk patterns
        return (
            keyword_count >= 2 or
            any(pattern in entity_name for pattern in [
                "sanctioned", "designated", "blocked", "prohibited"
            ])
        )
    
    def _update_state_with_hits(self, state: AMLState, hits: List[str]) -> AMLState:
        """Update state with sanctions screening results"""
        updated_state = state.copy()
        updated_state["sanction_hits"] = hits
        
        # Add risk factors for each type of hit
        risk_factors = []
        for hit in hits:
            if "DIRECT_MATCH" in hit:
                risk_factors.append("SANCTIONS_DIRECT_HIT")
            elif "KEYWORD_MATCH" in hit:
                risk_factors.append("SANCTIONS_KEYWORD_MATCH")
            elif "BENEFICIAL_OWNER" in hit:
                risk_factors.append("SANCTIONS_BENEFICIAL_OWNER")
            elif "PATTERN_MATCH" in hit:
                risk_factors.append("SANCTIONS_PATTERN_MATCH")
        
        return self.add_risk_factors(updated_state, risk_factors)
    
    def _generate_sanctions_alerts(self, state: AMLState, hits: List[str]) -> AMLState:
        """Generate appropriate alerts based on sanctions hits"""
        if not hits:
            return state
        
        transaction = state["transaction"]
        customer = state["customer"]
        
        # Determine alert severity based on hit types
        severity = RiskLevel.MEDIUM
        confidence = 0.7
        
        # Critical severity for direct hits
        direct_hits = [hit for hit in hits if "DIRECT_MATCH" in hit]
        if direct_hits:
            severity = RiskLevel.CRITICAL
            confidence = 0.95
        
        # High severity for beneficial owner hits
        beneficial_owner_hits = [hit for hit in hits if "BENEFICIAL_OWNER" in hit]
        if beneficial_owner_hits:
            severity = max(severity, RiskLevel.HIGH)
            confidence = max(confidence, 0.9)
        
        # Create comprehensive alert
        evidence = []
        for hit in hits:
            parts = hit.split(":")
            if len(parts) >= 3:
                match_type, entity, matched_item = parts[0], parts[1], parts[2]
                evidence.append(f"{match_type}: '{entity}' matched '{matched_item}'")
            else:
                evidence.append(hit)
        
        alert = self.create_alert(
            AlertType.SANCTIONS_HIT,
            f"Sanctions screening identified {len(hits)} potential matches",
            severity=severity,
            confidence=confidence,
            evidence=evidence,
            transaction_id=transaction.transaction_id,
            customer_id=customer.customer_id
        )
        
        return self.add_alert_to_state(state, alert)