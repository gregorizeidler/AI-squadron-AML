"""
Cryptocurrency Risk Analysis Agent for AML Detection System

This agent specializes in analyzing cryptocurrency-specific money laundering
risks including mixers, privacy coins, chain hopping, and darknet connections.
"""
from typing import List, Dict, Any
from decimal import Decimal

from .base_agent import BaseAgent
from ..core.state_manager import AMLState
from ..models.enums import AlertType, RiskLevel, AnalysisType
from ..services.llm_service import LLMService
from config.settings import DARKNET_MARKETS


class CryptoRiskAgent(BaseAgent):
    """
    Specialized agent for cryptocurrency transaction risk analysis.
    
    Responsibilities:
    - Analyze blockchain transaction patterns
    - Detect mixer and tumbler usage
    - Identify privacy coin transactions
    - Screen for darknet market connections
    - Assess wallet age and reputation
    - Detect chain-hopping patterns
    """
    
    def __init__(self, llm_service: LLMService):
        """Initialize the cryptocurrency risk agent"""
        super().__init__(llm_service, "crypto_specialist")
        
        # Known cryptocurrency mixers and tumblers
        self.known_mixers = {
            "tornado_cash", "wasabi_wallet", "samourai_whirlpool",
            "coinjoin", "joinmarket", "blender_io", "helix_mixer",
            "bitcoin_mixer", "ethereum_mixer", "anonymixer"
        }
        
        # Privacy-focused cryptocurrencies
        self.privacy_coins = {
            "XMR": "Monero",
            "ZEC": "Zcash", 
            "DASH": "Dash",
            "XVG": "Verge",
            "BEAM": "Beam",
            "GRIN": "Grin",
            "ZEN": "Horizen",
            "OXEN": "Oxen"
        }
        
        # High-risk exchanges and services
        self.high_risk_exchanges = {
            "p2p_exchange", "unregulated_exchange", "darknet_exchange",
            "mixer_service", "tumbler_service", "no_kyc_exchange"
        }
        
        # Darknet markets from config
        self.darknet_markets = set(DARKNET_MARKETS)
    
    def analyze(self, state: AMLState) -> AMLState:
        """
        Perform comprehensive cryptocurrency risk analysis.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with crypto risk analysis results
        """
        return self.analyze_crypto_risks(state)
    
    def analyze_crypto_risks(self, state: AMLState) -> AMLState:
        """
        Analyze cryptocurrency-specific money laundering risks.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with crypto analysis results
        """
        if not self.validate_state(state):
            return state
        
        transaction = state["transaction"]
        
        # Only analyze if this is a crypto transaction
        if transaction.asset_type.value != "CRYPTO":
            return state
        
        self.log_analysis_start(state, "crypto_risk_analysis")
        
        # Update path tracking
        state = self.update_path(state, "crypto_analysis")
        
        try:
            # Check if crypto details are available
            if not transaction.crypto_details:
                return self._handle_missing_crypto_details(state)
            
            crypto_details = transaction.crypto_details
            
            # Perform various crypto-specific risk assessments
            state = self._analyze_mixer_usage(state, crypto_details)
            state = self._analyze_wallet_characteristics(state, crypto_details)
            state = self._analyze_blockchain_patterns(state, crypto_details)
            state = self._analyze_darknet_connections(state, crypto_details)
            state = self._analyze_exchange_risks(state, crypto_details)
            state = self._analyze_privacy_features(state, crypto_details)
            
            # Generate comprehensive crypto risk assessment
            state = self._generate_crypto_risk_assessment(state)
            
            self.log_analysis_complete(
                state,
                "crypto_risk_analysis", 
                findings_count=len([rf for rf in state["risk_factors"] if "CRYPTO" in rf]),
                alerts_generated=len([a for a in state["alerts"] if "CRYPTO" in a.description])
            )
            
            return state
            
        except Exception as e:
            self.log_error(state, f"Error in crypto risk analysis: {str(e)}", e)
            return state
    
    def _handle_missing_crypto_details(self, state: AMLState) -> AMLState:
        """Handle cases where crypto details are missing"""
        alert = self.create_alert(
            AlertType.MISSING_DOCUMENTS,
            "Cryptocurrency transaction missing detailed blockchain information",
            severity=RiskLevel.MEDIUM,
            confidence=0.8,
            transaction_id=state["transaction"].transaction_id,
            customer_id=state["customer"].customer_id
        )
        
        state = self.add_alert_to_state(state, alert)
        return self.add_risk_factors(state, ["CRYPTO_MISSING_DETAILS"])
    
    def _analyze_mixer_usage(self, state: AMLState, crypto_details) -> AMLState:
        """Analyze cryptocurrency mixer and tumbler usage"""
        risks = []
        alerts = []
        
        # Direct mixer usage detection
        if crypto_details.mixer_used:
            risks.append("CRYPTO_MIXER_USED")
            
            alert = self.create_alert(
                AlertType.CRYPTO_MIXER,
                "Transaction involved cryptocurrency mixer or tumbler service",
                severity=RiskLevel.HIGH,
                confidence=0.95,
                evidence=["Mixer usage detected in transaction metadata"],
                transaction_id=state["transaction"].transaction_id,
                customer_id=state["customer"].customer_id
            )
            alerts.append(alert)
        
        # Check for known mixer services in metadata
        transaction_data = str(crypto_details.dict()).lower()
        for mixer in self.known_mixers:
            if mixer in transaction_data:
                risks.append(f"CRYPTO_KNOWN_MIXER_{mixer.upper()}")
                
                alert = self.create_alert(
                    AlertType.CRYPTO_MIXER,
                    f"Transaction associated with known mixer service: {mixer}",
                    severity=RiskLevel.HIGH,
                    confidence=0.9,
                    evidence=[f"Reference to {mixer} found in transaction data"],
                    transaction_id=state["transaction"].transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _analyze_wallet_characteristics(self, state: AMLState, crypto_details) -> AMLState:
        """Analyze wallet age and characteristics"""
        risks = []
        alerts = []
        
        # New wallet risk
        if crypto_details.wallet_age_days is not None:
            if crypto_details.wallet_age_days < 1:
                risks.append("CRYPTO_BRAND_NEW_WALLET")
                
                alert = self.create_alert(
                    AlertType.UNUSUAL_PATTERN,
                    f"Transaction from brand new wallet (age: {crypto_details.wallet_age_days} days)",
                    severity=RiskLevel.HIGH,
                    confidence=0.8,
                    transaction_id=state["transaction"].transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts.append(alert)
                
            elif crypto_details.wallet_age_days < 7:
                risks.append("CRYPTO_NEW_WALLET")
                
                alert = self.create_alert(
                    AlertType.UNUSUAL_PATTERN,
                    f"Transaction from very new wallet (age: {crypto_details.wallet_age_days} days)",
                    severity=RiskLevel.MEDIUM,
                    confidence=0.7,
                    transaction_id=state["transaction"].transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _analyze_blockchain_patterns(self, state: AMLState, crypto_details) -> AMLState:
        """Analyze blockchain transaction patterns"""
        risks = []
        alerts = []
        
        # Chain hopping detection
        if crypto_details.cross_chain_swaps > 0:
            if crypto_details.cross_chain_swaps >= 5:
                risks.append("CRYPTO_EXCESSIVE_CHAIN_HOPPING")
                
                alert = self.create_alert(
                    AlertType.UNUSUAL_PATTERN,
                    f"Excessive cross-chain swaps detected ({crypto_details.cross_chain_swaps} swaps)",
                    severity=RiskLevel.HIGH,
                    confidence=0.9,
                    evidence=[f"Transaction involved {crypto_details.cross_chain_swaps} cross-chain swaps"],
                    transaction_id=state["transaction"].transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts.append(alert)
                
            elif crypto_details.cross_chain_swaps >= 3:
                risks.append("CRYPTO_CHAIN_HOPPING")
                
                alert = self.create_alert(
                    AlertType.UNUSUAL_PATTERN,
                    f"Multiple cross-chain swaps detected ({crypto_details.cross_chain_swaps} swaps)",
                    severity=RiskLevel.MEDIUM,
                    confidence=0.8,
                    transaction_id=state["transaction"].transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _analyze_darknet_connections(self, state: AMLState, crypto_details) -> AMLState:
        """Analyze connections to darknet markets"""
        risks = []
        alerts = []
        
        # Direct darknet market connection
        if crypto_details.darknet_market:
            darknet_market = crypto_details.darknet_market.lower()
            
            if darknet_market in [dm.lower() for dm in self.darknet_markets]:
                risks.append("CRYPTO_KNOWN_DARKNET_MARKET")
                
                alert = self.create_alert(
                    AlertType.DARKNET_CONNECTION,
                    f"Transaction connected to known darknet market: {crypto_details.darknet_market}",
                    severity=RiskLevel.CRITICAL,
                    confidence=0.95,
                    evidence=[f"Direct connection to {crypto_details.darknet_market}"],
                    transaction_id=state["transaction"].transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts.append(alert)
            else:
                risks.append("CRYPTO_SUSPECTED_DARKNET_CONNECTION")
                
                alert = self.create_alert(
                    AlertType.DARKNET_CONNECTION,
                    f"Transaction potentially connected to darknet market: {crypto_details.darknet_market}",
                    severity=RiskLevel.HIGH,
                    confidence=0.8,
                    transaction_id=state["transaction"].transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _analyze_exchange_risks(self, state: AMLState, crypto_details) -> AMLState:
        """Analyze cryptocurrency exchange risks"""
        risks = []
        alerts = []
        
        if crypto_details.exchange_used:
            exchange = crypto_details.exchange_used.lower()
            
            # Check for high-risk exchanges
            if exchange in self.high_risk_exchanges:
                risks.append("CRYPTO_HIGH_RISK_EXCHANGE")
                
                alert = self.create_alert(
                    AlertType.HIGH_RISK_COUNTRY,  # Reusing this enum for high-risk services
                    f"Transaction involved high-risk exchange: {crypto_details.exchange_used}",
                    severity=RiskLevel.HIGH,
                    confidence=0.9,
                    transaction_id=state["transaction"].transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _analyze_privacy_features(self, state: AMLState, crypto_details) -> AMLState:
        """Analyze privacy-focused cryptocurrency features"""
        risks = []
        alerts = []
        
        # Privacy coin detection
        if crypto_details.privacy_coin or (
            crypto_details.cryptocurrency and 
            crypto_details.cryptocurrency.upper() in self.privacy_coins
        ):
            risks.append("CRYPTO_PRIVACY_COIN")
            
            coin_name = self.privacy_coins.get(
                crypto_details.cryptocurrency.upper() if crypto_details.cryptocurrency else "Unknown",
                crypto_details.cryptocurrency or "Unknown"
            )
            
            alert = self.create_alert(
                AlertType.UNUSUAL_PATTERN,
                f"Transaction involves privacy-focused cryptocurrency: {coin_name}",
                severity=RiskLevel.MEDIUM,
                confidence=0.8,
                evidence=[f"Privacy coin detected: {coin_name}"],
                transaction_id=state["transaction"].transaction_id,
                customer_id=state["customer"].customer_id
            )
            alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _generate_crypto_risk_assessment(self, state: AMLState) -> AMLState:
        """Generate comprehensive cryptocurrency risk assessment"""
        crypto_risks = [rf for rf in state["risk_factors"] if "CRYPTO" in rf]
        
        if not crypto_risks:
            return state
        
        # Create LLM analysis for comprehensive crypto assessment
        analysis_prompt = f"""
        Analyze the following cryptocurrency transaction risks and provide a comprehensive assessment:
        
        Risk Factors: {crypto_risks}
        Transaction Amount: {state['transaction'].amount} {state['transaction'].currency}
        Crypto Details: {state['transaction'].crypto_details.dict() if state['transaction'].crypto_details else 'Limited'}
        
        Provide assessment covering:
        1. Overall risk level for money laundering
        2. Specific crypto-related concerns
        3. Recommended monitoring actions
        4. Compliance considerations
        
        Focus on practical AML implications and enforcement priorities.
        """
        
        try:
            # Get LLM analysis
            llm_response = self.llm_service.analyze_text(
                analysis_prompt,
                analysis_type="crypto_risk_assessment"
            )
            
            # Extract risk indicators from LLM response
            risk_indicators = self.extract_risk_indicators_from_text(llm_response)
            
            # Create LLM analysis result
            llm_analysis = self.create_llm_analysis_result(
                analysis_type=AnalysisType.CRYPTO_ANALYSIS,
                analysis_text=llm_response,
                key_findings=crypto_risks,
                risk_indicators=risk_indicators,
                confidence_score=0.85,
                model_used=self.llm_service.get_model_name()
            )
            
            # Add to state
            state = self.add_llm_analysis_to_state(state, llm_analysis)
            
        except Exception as e:
            self.log_error(state, f"Error in LLM crypto analysis: {str(e)}", e)
        
        return state