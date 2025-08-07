"""
Main AML Detection System

This is the primary entry point for the AI Squadron AML Detection System.
It coordinates all components and provides a high-level interface for
transaction analysis using a squadron of specialized AI agents.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..core.workflow_engine import WorkflowEngine
from ..core.state_manager import StateManager
from ..models.transaction import Transaction
from ..models.customer import Customer
from ..models.analysis import AMLAnalysisResult
from ..services.monitoring_service import MonitoringService
from ..services.notification_service import NotificationService
from ..services.persistence_service import PersistenceService
from config.settings import settings


class AMLSystem:
    """
    Main AML Detection System implementing AI Squadron coordination principles.
    
    This system embodies the seven core principles of ambient intelligence
    through coordinated AI agent collaboration:
    1. Goal-oriented: Focused on detecting suspicious activities while minimizing false positives
    2. Autonomous operation: Makes decisions independently with appropriate human oversight
    3. Continuous perception: Monitors transaction streams and external events in real-time
    4. Semantic reasoning: Uses LLMs for contextual understanding beyond rule matching
    5. Persistence across interactions: Maintains context and learns from past analyses
    6. Multi-agent collaboration: Coordinates specialized agents for comprehensive analysis
    7. Asynchronous communication: Event-driven architecture for scalable processing
    """
    
    def __init__(self):
        """Initialize the AML system with all necessary components"""
        self.logger = self._setup_logging()
        
        # Core components
        self.workflow_engine = WorkflowEngine()
        self.state_manager = StateManager()
        
        # Services
        self.monitoring_service = MonitoringService()
        self.notification_service = NotificationService()
        self.persistence_service = PersistenceService()
        
        # System state
        self.is_running = False
        self.processed_count = 0
        self.start_time: Optional[datetime] = None
        
        self.logger.info("AML System initialized successfully")
    
    def start(self) -> None:
        """Start the AML system and initialize background services"""
        if self.is_running:
            self.logger.warning("AML System is already running")
            return
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        # Start background services
        self.monitoring_service.start()
        
        self.logger.info("AML System started successfully")
    
    def stop(self) -> None:
        """Stop the AML system and cleanup resources"""
        if not self.is_running:
            self.logger.warning("AML System is not running")
            return
        
        self.is_running = False
        
        # Stop background services
        self.monitoring_service.stop()
        
        self.logger.info("AML System stopped successfully")
    
    async def analyze_transaction_async(
        self,
        transaction: Transaction,
        customer: Customer,
        context: Optional[Dict[str, Any]] = None
    ) -> AMLAnalysisResult:
        """
        Asynchronously analyze a transaction for AML risks.
        
        Args:
            transaction: Transaction to analyze
            customer: Customer associated with the transaction
            context: Additional context for the analysis
            
        Returns:
            Complete AML analysis result
        """
        if not self.is_running:
            raise RuntimeError("AML System is not running. Call start() first.")
        
        start_time = datetime.utcnow()
        
        try:
            # Create initial state
            initial_state = self.state_manager.create_initial_state(
                transaction=transaction,
                customer=customer
            )
            
            # Add any additional context
            if context:
                initial_state["metadata"].update(context)
            
            # Log the start of analysis
            self.logger.info(
                f"Starting AML analysis for transaction {transaction.transaction_id}",
                extra={
                    "transaction_id": transaction.transaction_id,
                    "customer_id": customer.customer_id,
                    "amount": str(transaction.amount),
                    "currency": transaction.currency
                }
            )
            
            # Execute the workflow
            final_state = await self.workflow_engine.analyze_transaction_async(initial_state)
            
            # Convert to analysis result
            result = self.state_manager.convert_to_analysis_result(final_state)
            
            # Update processing metadata
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            result.processing_time_seconds = processing_time
            result.processing_completed = datetime.utcnow()
            
            # Persist the result
            await self.persistence_service.save_analysis_result(result)
            
            # Send notifications if required
            await self._handle_notifications(result)
            
            # Update monitoring metrics
            self.monitoring_service.record_analysis_completed(result)
            self.processed_count += 1
            
            # Log completion
            self.logger.info(
                f"AML analysis completed for transaction {transaction.transaction_id}",
                extra={
                    "analysis_id": result.analysis_id,
                    "risk_score": result.risk_assessment.risk_score,
                    "final_status": result.final_status,
                    "processing_time": processing_time,
                    "requires_review": result.requires_human_review,
                    "sar_recommended": result.sar_recommended
                }
            )
            
            return result
            
        except Exception as e:
            # Log the error
            self.logger.error(
                f"Error analyzing transaction {transaction.transaction_id}: {str(e)}",
                extra={
                    "transaction_id": transaction.transaction_id,
                    "customer_id": customer.customer_id,
                    "error": str(e)
                },
                exc_info=True
            )
            
            # Create error result
            error_result = self._create_error_result(transaction, customer, str(e))
            
            # Record error metrics
            self.monitoring_service.record_analysis_error(transaction.transaction_id, str(e))
            
            return error_result
    
    def analyze_transaction(
        self,
        transaction: Transaction,
        customer: Customer,
        context: Optional[Dict[str, Any]] = None
    ) -> AMLAnalysisResult:
        """
        Synchronously analyze a transaction for AML risks.
        
        Args:
            transaction: Transaction to analyze
            customer: Customer associated with the transaction
            context: Additional context for the analysis
            
        Returns:
            Complete AML analysis result
        """
        return asyncio.run(self.analyze_transaction_async(transaction, customer, context))
    
    async def analyze_batch_async(
        self,
        transactions: List[tuple[Transaction, Customer]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[AMLAnalysisResult]:
        """
        Asynchronously analyze a batch of transactions.
        
        Args:
            transactions: List of (transaction, customer) tuples to analyze
            context: Additional context for the analyses
            
        Returns:
            List of AML analysis results
        """
        tasks = []
        
        for transaction, customer in transactions:
            task = self.analyze_transaction_async(transaction, customer, context)
            tasks.append(task)
        
        # Execute all analyses concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                transaction, customer = transactions[i]
                error_result = self._create_error_result(transaction, customer, str(result))
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and metrics.
        
        Returns:
            Dictionary containing system status information
        """
        uptime = None
        if self.start_time:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": uptime,
            "processed_transactions": self.processed_count,
            "monitoring_metrics": self.monitoring_service.get_metrics(),
            "workflow_info": {
                "workflow_version": "1.0",
                "available_agents": [
                    "orchestrator", "sanctions", "pep", "geographic",
                    "behavioral", "crypto", "document", "enhanced_dd"
                ]
            }
        }
    
    def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get summary of recent analyses.
        
        Args:
            limit: Maximum number of recent analyses to return
            
        Returns:
            List of analysis summaries
        """
        return self.persistence_service.get_recent_analyses(limit)
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[AMLAnalysisResult]:
        """
        Retrieve a specific analysis by its ID.
        
        Args:
            analysis_id: Unique analysis identifier
            
        Returns:
            AML analysis result or None if not found
        """
        return self.persistence_service.get_analysis_by_id(analysis_id)
    
    def get_customer_risk_profile(self, customer_id: str) -> Dict[str, Any]:
        """
        Get comprehensive risk profile for a customer.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Customer risk profile including historical analysis
        """
        return self.persistence_service.get_customer_risk_profile(customer_id)
    
    async def _handle_notifications(self, result: AMLAnalysisResult) -> None:
        """Handle notifications based on analysis results"""
        try:
            # High-risk notifications
            if result.sar_recommended:
                await self.notification_service.send_sar_notification(result)
            
            # Human review notifications  
            if result.requires_human_review:
                await self.notification_service.send_review_notification(result)
            
            # Critical alert notifications
            critical_alerts = [
                alert for alert in result.alerts 
                if alert.severity.value == "CRITICAL"
            ]
            
            if critical_alerts:
                await self.notification_service.send_critical_alert_notification(
                    result, critical_alerts
                )
                
        except Exception as e:
            self.logger.error(f"Error sending notifications: {str(e)}", exc_info=True)
    
    def _create_error_result(
        self,
        transaction: Transaction,
        customer: Customer,
        error_message: str
    ) -> AMLAnalysisResult:
        """Create an error analysis result"""
        from ..models.analysis import RiskAssessment
        from ..models.enums import RiskLevel
        
        # Create minimal risk assessment for error case
        error_risk_assessment = RiskAssessment(
            risk_score=0,
            risk_level=RiskLevel.LOW,
            recommended_action="SYSTEM_ERROR_REVIEW_REQUIRED"
        )
        
        return AMLAnalysisResult(
            analysis_id=f"ERROR_{transaction.transaction_id}_{int(datetime.utcnow().timestamp())}",
            transaction_id=transaction.transaction_id,
            customer_id=customer.customer_id,
            decision_path=["error"],
            risk_assessment=error_risk_assessment,
            final_status="ERROR",
            requires_human_review=True,
            sar_recommended=False,
            transaction_approved=False,
            processing_started=datetime.utcnow(),
            metadata={"error_message": error_message}
        )
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for the AML system"""
        logger = logging.getLogger("aml_system")
        logger.setLevel(getattr(logging, settings.system.log_level))
        
        # Create console handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()