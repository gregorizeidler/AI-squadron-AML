"""
Behavioral Analysis Agent for AML Detection System

This agent specializes in analyzing behavioral patterns and transaction
behaviors that may indicate money laundering or suspicious activities.
"""
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from decimal import Decimal

from .base_agent import BaseAgent
from ..core.state_manager import AMLState
from ..models.enums import AlertType, RiskLevel, AnalysisType
from ..services.llm_service import LLMService


class BehavioralAnalysisAgent(BaseAgent):
    """
    Specialized agent for behavioral pattern analysis and anomaly detection.
    
    Responsibilities:
    - Detect structuring and smurfing patterns
    - Identify unusual transaction timing
    - Analyze transaction frequency patterns
    - Detect rapid fund movement
    - Assess customer behavior changes
    - Identify round-number transactions
    - Detect velocity and volume anomalies
    """
    
    def __init__(self, llm_service: LLMService):
        """Initialize the behavioral analysis agent"""
        super().__init__(llm_service, "behavioral_analyst")
        
        # Structuring detection thresholds
        self.structuring_threshold = Decimal("10000")  # CTR threshold
        self.micro_structuring_threshold = Decimal("3000")
        self.structuring_time_window_hours = 24
        self.max_daily_transactions = 5
        
        # Velocity thresholds
        self.rapid_velocity_threshold = 10  # transactions per hour
        self.high_velocity_threshold = 24   # transactions per day
        
        # Round number patterns
        self.round_number_bases = [1000, 5000, 10000, 25000, 50000, 100000]
        self.round_number_tolerance = 0.01  # 1% tolerance
        
        # Dormancy patterns
        self.dormancy_threshold_days = 90
        self.sudden_activity_multiplier = 5
    
    def analyze(self, state: AMLState) -> AMLState:
        """
        Perform comprehensive behavioral pattern analysis.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with behavioral analysis results
        """
        return self.analyze_patterns(state)
    
    def analyze_patterns(self, state: AMLState) -> AMLState:
        """
        Analyze behavioral patterns for suspicious activity.
        
        Args:
            state: Current AML analysis state
            
        Returns:
            Updated state with behavioral analysis results
        """
        if not self.validate_state(state):
            return state
        
        self.log_analysis_start(state, "behavioral_pattern_analysis")
        
        # Update path tracking
        state = self.update_path(state, "behavioral_analysis")
        
        try:
            transaction = state["transaction"]
            customer = state["customer"]
            
            # Get customer transaction history
            transaction_history = self._get_transaction_history(customer)
            
            # Perform various behavioral analyses
            state = self._detect_structuring_patterns(state, transaction, transaction_history)
            state = self._analyze_transaction_timing(state, transaction, transaction_history)
            state = self._detect_velocity_anomalies(state, transaction, transaction_history)
            state = self._analyze_amount_patterns(state, transaction, transaction_history)
            state = self._detect_dormancy_patterns(state, customer, transaction_history)
            state = self._analyze_frequency_patterns(state, transaction, transaction_history)
            state = self._detect_round_numbers(state, transaction)
            
            # Generate comprehensive behavioral assessment
            state = self._generate_behavioral_assessment(state)
            
            self.log_analysis_complete(
                state,
                "behavioral_pattern_analysis",
                findings_count=len([rf for rf in state["risk_factors"] if "BEHAVIORAL" in rf or "STRUCTURING" in rf]),
                alerts_generated=len([a for a in state["alerts"] if "behavioral" in a.description.lower() or "pattern" in a.description.lower()])
            )
            
            return state
            
        except Exception as e:
            self.log_error(state, f"Error in behavioral analysis: {str(e)}", e)
            return state
    
    def _get_transaction_history(self, customer) -> List[Dict]:
        """Extract transaction history from customer data"""
        # Use the transaction_history from customer model
        if hasattr(customer, 'transaction_history') and customer.transaction_history:
            return customer.transaction_history
        
        # Fallback to generating mock history based on customer data
        return []
    
    def _detect_structuring_patterns(self, state: AMLState, transaction, history: List[Dict]) -> AMLState:
        """Detect structuring and smurfing patterns"""
        risks = []
        alerts = []
        
        current_time = transaction.timestamp
        
        # Get recent transactions within time window
        time_window = timedelta(hours=self.structuring_time_window_hours)
        recent_transactions = []
        
        for hist_tx in history:
            if hasattr(hist_tx, 'timestamp'):
                tx_time = hist_tx.timestamp
            else:
                tx_time = hist_tx.get('timestamp', current_time - timedelta(days=1))
            
            if current_time - tx_time <= time_window:
                tx_amount = hist_tx.get('amount', hist_tx.get('transaction_amount', 0))
                recent_transactions.append({
                    'amount': Decimal(str(tx_amount)),
                    'timestamp': tx_time
                })
        
        # Add current transaction
        recent_transactions.append({
            'amount': transaction.amount,
            'timestamp': current_time
        })
        
        # Classic structuring detection
        if len(recent_transactions) >= 3:
            # Check if all transactions are below reporting threshold
            below_threshold = all(tx['amount'] < self.structuring_threshold for tx in recent_transactions)
            
            if below_threshold and len(recent_transactions) >= self.max_daily_transactions:
                risks.append("CLASSIC_STRUCTURING")
                
                total_amount = sum(tx['amount'] for tx in recent_transactions)
                
                alert = self.create_alert(
                    AlertType.STRUCTURING,
                    f"Classic structuring pattern: {len(recent_transactions)} transactions totaling {total_amount} in {self.structuring_time_window_hours} hours",
                    severity=RiskLevel.HIGH,
                    confidence=0.9,
                    evidence=[
                        f"Transaction count: {len(recent_transactions)}",
                        f"Total amount: {total_amount}",
                        f"All amounts below {self.structuring_threshold}"
                    ],
                    transaction_id=transaction.transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts.append(alert)
        
        # Micro-structuring detection
        micro_transactions = [tx for tx in recent_transactions if tx['amount'] < self.micro_structuring_threshold]
        
        if len(micro_transactions) >= 5:
            risks.append("MICRO_STRUCTURING")
            
            alert = self.create_alert(
                AlertType.STRUCTURING,
                f"Micro-structuring pattern: {len(micro_transactions)} small transactions below {self.micro_structuring_threshold}",
                severity=RiskLevel.MEDIUM,
                confidence=0.8,
                transaction_id=transaction.transaction_id,
                customer_id=state["customer"].customer_id
            )
            alerts.append(alert)
        
        # Uniform amount structuring
        if len(recent_transactions) >= 3:
            amounts = [float(tx['amount']) for tx in recent_transactions]
            if statistics.stdev(amounts) < 500:  # Very similar amounts
                risks.append("UNIFORM_AMOUNT_STRUCTURING")
                
                alert = self.create_alert(
                    AlertType.STRUCTURING,
                    f"Uniform amount structuring: {len(recent_transactions)} transactions with similar amounts",
                    severity=RiskLevel.MEDIUM,
                    confidence=0.75,
                    evidence=[f"Amount standard deviation: {statistics.stdev(amounts):.2f}"],
                    transaction_id=transaction.transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _analyze_transaction_timing(self, state: AMLState, transaction, history: List[Dict]) -> AMLState:
        """Analyze transaction timing patterns"""
        risks = []
        alerts = []
        
        current_time = transaction.timestamp
        
        # Check for unusual timing
        if self._is_unusual_timing(current_time):
            risks.append("UNUSUAL_TIMING")
            
            alert = self.create_alert(
                AlertType.UNUSUAL_PATTERN,
                f"Transaction occurred at unusual time: {current_time.strftime('%H:%M on %A')}",
                severity=RiskLevel.LOW,
                confidence=0.6,
                transaction_id=transaction.transaction_id,
                customer_id=state["customer"].customer_id
            )
            alerts.append(alert)
        
        # Analyze timing intervals between transactions
        if len(history) >= 2:
            intervals = self._calculate_timing_intervals(history + [{'timestamp': current_time}])
            
            # Detect suspiciously regular intervals
            if self._has_regular_intervals(intervals):
                risks.append("REGULAR_TIMING_PATTERN")
                
                alert = self.create_alert(
                    AlertType.UNUSUAL_PATTERN,
                    "Suspiciously regular timing pattern detected",
                    severity=RiskLevel.MEDIUM,
                    confidence=0.7,
                    evidence=["Transactions occur at very regular intervals"],
                    transaction_id=transaction.transaction_id,
                    customer_id=state["customer"].customer_id
                )
                alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _detect_velocity_anomalies(self, state: AMLState, transaction, history: List[Dict]) -> AMLState:
        """Detect transaction velocity anomalies"""
        risks = []
        alerts = []
        
        current_time = transaction.timestamp
        
        # Count transactions in last hour
        hour_ago = current_time - timedelta(hours=1)
        recent_hour_count = len([
            tx for tx in history 
            if tx.get('timestamp', current_time - timedelta(days=1)) > hour_ago
        ]) + 1  # +1 for current transaction
        
        if recent_hour_count >= self.rapid_velocity_threshold:
            risks.append("RAPID_TRANSACTION_VELOCITY")
            
            alert = self.create_alert(
                AlertType.RAPID_MOVEMENT,
                f"Rapid transaction velocity: {recent_hour_count} transactions in the last hour",
                severity=RiskLevel.HIGH,
                confidence=0.9,
                transaction_id=transaction.transaction_id,
                customer_id=state["customer"].customer_id
            )
            alerts.append(alert)
        
        # Count transactions in last 24 hours
        day_ago = current_time - timedelta(hours=24)
        recent_day_count = len([
            tx for tx in history 
            if tx.get('timestamp', current_time - timedelta(days=1)) > day_ago
        ]) + 1
        
        if recent_day_count >= self.high_velocity_threshold:
            risks.append("HIGH_DAILY_VELOCITY")
            
            alert = self.create_alert(
                AlertType.RAPID_MOVEMENT,
                f"High daily transaction velocity: {recent_day_count} transactions in 24 hours",
                severity=RiskLevel.MEDIUM,
                confidence=0.8,
                transaction_id=transaction.transaction_id,
                customer_id=state["customer"].customer_id
            )
            alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _analyze_amount_patterns(self, state: AMLState, transaction, history: List[Dict]) -> AMLState:
        """Analyze transaction amount patterns"""
        risks = []
        alerts = []
        
        if not history:
            return state
        
        # Get historical amounts
        historical_amounts = [
            Decimal(str(tx.get('amount', tx.get('transaction_amount', 0))))
            for tx in history
        ]
        
        if not historical_amounts:
            return state
        
        current_amount = transaction.amount
        avg_amount = sum(historical_amounts) / len(historical_amounts)
        max_amount = max(historical_amounts)
        
        # Detect amount anomalies
        if current_amount > avg_amount * 10:  # 10x average
            risks.append("AMOUNT_ANOMALY_HIGH")
            
            alert = self.create_alert(
                AlertType.UNUSUAL_PATTERN,
                f"Transaction amount ({current_amount}) significantly higher than average ({avg_amount:.2f})",
                severity=RiskLevel.MEDIUM,
                confidence=0.8,
                transaction_id=transaction.transaction_id,
                customer_id=state["customer"].customer_id
            )
            alerts.append(alert)
        
        # Check for sudden amount escalation
        if current_amount > max_amount * 2:  # 2x previous maximum
            risks.append("SUDDEN_AMOUNT_ESCALATION")
            
            alert = self.create_alert(
                AlertType.UNUSUAL_PATTERN,
                f"Sudden transaction amount escalation: {current_amount} vs previous max {max_amount}",
                severity=RiskLevel.MEDIUM,
                confidence=0.7,
                transaction_id=transaction.transaction_id,
                customer_id=state["customer"].customer_id
            )
            alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _detect_dormancy_patterns(self, state: AMLState, customer, history: List[Dict]) -> AMLState:
        """Detect dormant account activation patterns"""
        risks = []
        alerts = []
        
        # Check if account was dormant
        if customer.account_age_days > self.dormancy_threshold_days and len(history) == 0:
            risks.append("DORMANT_ACCOUNT_ACTIVATION")
            
            alert = self.create_alert(
                AlertType.UNUSUAL_PATTERN,
                f"Dormant account activation: Account age {customer.account_age_days} days with no recent activity",
                severity=RiskLevel.MEDIUM,
                confidence=0.8,
                transaction_id=state["transaction"].transaction_id,
                customer_id=customer.customer_id
            )
            alerts.append(alert)
        
        # Check for sudden activity after dormancy
        elif history:
            # Find the most recent transaction before current
            recent_dates = [
                tx.get('timestamp', state["transaction"].timestamp - timedelta(days=365))
                for tx in history
            ]
            
            if recent_dates:
                last_activity = max(recent_dates)
                days_since_last = (state["transaction"].timestamp - last_activity).days
                
                if days_since_last > self.dormancy_threshold_days:
                    risks.append("SUDDEN_ACTIVITY_AFTER_DORMANCY")
                    
                    alert = self.create_alert(
                        AlertType.UNUSUAL_PATTERN,
                        f"Sudden activity after {days_since_last} days of dormancy",
                        severity=RiskLevel.MEDIUM,
                        confidence=0.75,
                        transaction_id=state["transaction"].transaction_id,
                        customer_id=customer.customer_id
                    )
                    alerts.append(alert)
        
        # Update state
        updated_state = self.add_risk_factors(state, risks)
        for alert in alerts:
            updated_state = self.add_alert_to_state(updated_state, alert)
        
        return updated_state
    
    def _analyze_frequency_patterns(self, state: AMLState, transaction, history: List[Dict]) -> AMLState:
        """Analyze transaction frequency patterns"""
        risks = []
        
        # Calculate frequency metrics
        total_transactions = len(history) + 1  # Include current
        
        if hasattr(state["customer"], 'account_age_days'):
            account_age_days = max(state["customer"].account_age_days, 1)
            frequency_per_day = total_transactions / account_age_days
            
            # High frequency detection
            if frequency_per_day > 5:  # More than 5 transactions per day on average
                risks.append("HIGH_FREQUENCY_PATTERN")
            elif frequency_per_day > 2:
                risks.append("ELEVATED_FREQUENCY_PATTERN")
        
        # Pattern frequency (specific to current transaction pattern)
        if hasattr(transaction, 'frequency') and transaction.frequency > 5:
            risks.append("HIGH_PATTERN_FREQUENCY")
        
        return self.add_risk_factors(state, risks)
    
    def _detect_round_numbers(self, state: AMLState, transaction) -> AMLState:
        """Detect round number transactions"""
        risks = []
        
        if self._is_round_number(transaction.amount):
            risks.append("ROUND_NUMBER_TRANSACTION")
        
        return self.add_risk_factors(state, risks)
    
    def _generate_behavioral_assessment(self, state: AMLState) -> AMLState:
        """Generate comprehensive behavioral assessment using LLM"""
        behavioral_risks = [rf for rf in state["risk_factors"] if any(
            pattern in rf for pattern in [
                "BEHAVIORAL", "STRUCTURING", "VELOCITY", "TIMING", 
                "PATTERN", "ANOMALY", "DORMANT", "FREQUENCY"
            ]
        )]
        
        if not behavioral_risks:
            return state
        
        assessment_prompt = f"""
        Analyze the following behavioral patterns for AML risk assessment:
        
        Customer Account Age: {state['customer'].account_age_days} days
        Transaction Amount: {state['transaction'].amount} {state['transaction'].currency}
        Transaction History: {len(getattr(state['customer'], 'transaction_history', []))} previous transactions
        Behavioral Risk Factors: {behavioral_risks}
        
        Assessment should cover:
        1. Overall behavioral risk level
        2. Specific pattern concerns
        3. Money laundering methodology indicators
        4. Customer risk profile changes
        5. Monitoring and investigation recommendations
        
        Focus on practical implications for AML compliance and investigation.
        """
        
        try:
            response = self.llm_service.analyze_text(
                assessment_prompt,
                analysis_type="behavioral_risk_assessment"
            )
            
            # Extract additional risk indicators
            assessment_risks = self.extract_risk_indicators_from_text(response)
            
            # Create LLM analysis result
            llm_analysis = self.create_llm_analysis_result(
                analysis_type=AnalysisType.BEHAVIORAL_ANALYSIS,
                analysis_text=response,
                key_findings=behavioral_risks,
                risk_indicators=assessment_risks,
                confidence_score=0.85,
                model_used=self.llm_service.get_model_name()
            )
            
            # Add to state
            state = self.add_llm_analysis_to_state(state, llm_analysis)
            
        except Exception as e:
            self.log_error(state, f"Error in behavioral assessment: {str(e)}", e)
        
        return state
    
    # Helper methods
    
    def _is_unusual_timing(self, timestamp: datetime) -> bool:
        """Check if transaction timing is unusual"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Outside business hours (before 6 AM or after 10 PM)
        if hour < 6 or hour > 22:
            return True
        
        # Weekend transactions
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return True
        
        return False
    
    def _calculate_timing_intervals(self, transactions: List[Dict]) -> List[float]:
        """Calculate intervals between transactions in hours"""
        if len(transactions) < 2:
            return []
        
        intervals = []
        for i in range(1, len(transactions)):
            prev_time = transactions[i-1].get('timestamp')
            curr_time = transactions[i].get('timestamp')
            
            if prev_time and curr_time:
                interval = (curr_time - prev_time).total_seconds() / 3600  # Convert to hours
                intervals.append(interval)
        
        return intervals
    
    def _has_regular_intervals(self, intervals: List[float]) -> bool:
        """Check if intervals are suspiciously regular"""
        if len(intervals) < 3:
            return False
        
        # Check if intervals are very similar (within 10% variance)
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        coefficient_of_variation = (variance ** 0.5) / avg_interval
        
        return coefficient_of_variation < 0.1  # Less than 10% variation
    
    def _is_round_number(self, amount: Decimal) -> bool:
        """Check if amount is a round number"""
        amount_float = float(amount)
        
        for base in self.round_number_bases:
            if amount_float % base == 0:
                return True
            
            # Check within tolerance
            remainder = amount_float % base
            if remainder <= base * self.round_number_tolerance or remainder >= base * (1 - self.round_number_tolerance):
                return True
        
        return False