"""
Notification Service for AML Detection System

This service handles notifications for various AML events including
SAR generation, critical alerts, and review requirements.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models.analysis import AMLAnalysisResult, AlertDetails


class NotificationService:
    """
    Service for handling AML system notifications.
    
    Supports:
    - SAR generation notifications
    - Critical alert notifications  
    - Human review notifications
    - System health alerts
    - Regulatory reporting notifications
    """
    
    def __init__(self):
        """Initialize the notification service"""
        self.logger = self._setup_logging()
        self.notification_channels = {
            "email": EmailNotificationChannel(),
            "slack": SlackNotificationChannel(),
            "webhook": WebhookNotificationChannel(),
            "sms": SMSNotificationChannel()
        }
        
        # Notification routing rules
        self.routing_rules = {
            "sar_generation": ["email", "webhook"],
            "critical_alert": ["email", "slack", "sms"],
            "human_review": ["email", "slack"],
            "system_health": ["slack", "webhook"]
        }
    
    async def send_sar_notification(self, analysis_result: AMLAnalysisResult):
        """Send notification for SAR generation"""
        if not analysis_result.sar_recommended:
            return
        
        try:
            notification_data = {
                "type": "SAR_GENERATION",
                "analysis_id": analysis_result.analysis_id,
                "transaction_id": analysis_result.transaction_id,
                "customer_id": analysis_result.customer_id,
                "risk_score": analysis_result.risk_assessment.risk_score,
                "case_id": analysis_result.case_id,
                "timestamp": datetime.utcnow().isoformat(),
                "urgency": "HIGH"
            }
            
            message = self._format_sar_message(analysis_result)
            
            # Send to configured channels
            channels = self.routing_rules.get("sar_generation", [])
            await self._send_to_channels(channels, "SAR Generation Required", message, notification_data)
            
            self.logger.info(f"SAR notification sent for analysis {analysis_result.analysis_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending SAR notification: {str(e)}")
    
    async def send_critical_alert_notification(
        self, 
        analysis_result: AMLAnalysisResult, 
        critical_alerts: List[AlertDetails]
    ):
        """Send notification for critical alerts"""
        if not critical_alerts:
            return
        
        try:
            notification_data = {
                "type": "CRITICAL_ALERT",
                "analysis_id": analysis_result.analysis_id,
                "transaction_id": analysis_result.transaction_id,
                "customer_id": analysis_result.customer_id,
                "alert_count": len(critical_alerts),
                "alert_types": [alert.alert_type.value for alert in critical_alerts],
                "timestamp": datetime.utcnow().isoformat(),
                "urgency": "CRITICAL"
            }
            
            message = self._format_critical_alert_message(analysis_result, critical_alerts)
            
            # Send to configured channels
            channels = self.routing_rules.get("critical_alert", [])
            await self._send_to_channels(channels, "Critical AML Alert", message, notification_data)
            
            self.logger.info(f"Critical alert notification sent for analysis {analysis_result.analysis_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending critical alert notification: {str(e)}")
    
    async def send_review_notification(self, analysis_result: AMLAnalysisResult):
        """Send notification for human review requirement"""
        if not analysis_result.requires_human_review:
            return
        
        try:
            notification_data = {
                "type": "HUMAN_REVIEW",
                "analysis_id": analysis_result.analysis_id,
                "transaction_id": analysis_result.transaction_id,
                "customer_id": analysis_result.customer_id,
                "risk_score": analysis_result.risk_assessment.risk_score,
                "review_deadline": (datetime.utcnow().replace(hour=23, minute=59, second=59)).isoformat(),
                "timestamp": datetime.utcnow().isoformat(),
                "urgency": "MEDIUM"
            }
            
            message = self._format_review_message(analysis_result)
            
            # Send to configured channels
            channels = self.routing_rules.get("human_review", [])
            await self._send_to_channels(channels, "Human Review Required", message, notification_data)
            
            self.logger.info(f"Review notification sent for analysis {analysis_result.analysis_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending review notification: {str(e)}")
    
    async def send_system_health_notification(self, health_data: Dict[str, Any]):
        """Send system health notification"""
        if health_data.get("status") == "HEALTHY":
            return  # Don't notify for healthy status
        
        try:
            notification_data = {
                "type": "SYSTEM_HEALTH",
                "status": health_data.get("status"),
                "concerns": health_data.get("concerns", []),
                "error_rate": health_data.get("error_rate", 0),
                "timestamp": datetime.utcnow().isoformat(),
                "urgency": "LOW" if health_data.get("status") == "SLOW" else "MEDIUM"
            }
            
            message = self._format_health_message(health_data)
            
            # Send to configured channels
            channels = self.routing_rules.get("system_health", [])
            await self._send_to_channels(channels, "AML System Health Alert", message, notification_data)
            
            self.logger.warning(f"System health notification sent: {health_data.get('status')}")
            
        except Exception as e:
            self.logger.error(f"Error sending system health notification: {str(e)}")
    
    async def _send_to_channels(
        self, 
        channels: List[str], 
        subject: str, 
        message: str, 
        data: Dict[str, Any]
    ):
        """Send notification to multiple channels"""
        tasks = []
        
        for channel_name in channels:
            channel = self.notification_channels.get(channel_name)
            if channel and channel.is_configured():
                task = channel.send_notification(subject, message, data)
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _format_sar_message(self, analysis_result: AMLAnalysisResult) -> str:
        """Format SAR generation message"""
        return f"""
🚨 SUSPICIOUS ACTIVITY REPORT (SAR) GENERATION REQUIRED

Analysis ID: {analysis_result.analysis_id}
Transaction ID: {analysis_result.transaction_id}
Customer ID: {analysis_result.customer_id}
Risk Score: {analysis_result.risk_assessment.risk_score}/100

Key Risk Factors:
{self._format_risk_factors(analysis_result.risk_assessment.risk_factors)}

Recommended Action: {analysis_result.risk_assessment.recommended_action}

This transaction requires immediate SAR filing with regulatory authorities.
Case ID: {analysis_result.case_id}

Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    def _format_critical_alert_message(
        self, 
        analysis_result: AMLAnalysisResult, 
        alerts: List[AlertDetails]
    ) -> str:
        """Format critical alert message"""
        alert_details = []
        for alert in alerts:
            alert_details.append(f"- {alert.alert_type.value}: {alert.description}")
        
        return f"""
⚠️ CRITICAL AML ALERTS DETECTED

Analysis ID: {analysis_result.analysis_id}
Transaction ID: {analysis_result.transaction_id}
Customer ID: {analysis_result.customer_id}
Risk Score: {analysis_result.risk_assessment.risk_score}/100

Critical Alerts ({len(alerts)}):
{chr(10).join(alert_details)}

This transaction requires immediate attention and investigation.

Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    def _format_review_message(self, analysis_result: AMLAnalysisResult) -> str:
        """Format human review message"""
        return f"""
👤 HUMAN REVIEW REQUIRED

Analysis ID: {analysis_result.analysis_id}
Transaction ID: {analysis_result.transaction_id}
Customer ID: {analysis_result.customer_id}
Risk Score: {analysis_result.risk_assessment.risk_score}/100

Review Reason: Medium risk score requires analyst review
Decision Path: {' → '.join(analysis_result.decision_path)}

Please review this transaction within 24 hours.

Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    def _format_health_message(self, health_data: Dict[str, Any]) -> str:
        """Format system health message"""
        concerns = health_data.get("concerns", [])
        concern_list = "\n".join([f"- {concern}" for concern in concerns])
        
        return f"""
🔧 AML SYSTEM HEALTH ALERT

System Status: {health_data.get("status", "UNKNOWN")}
Error Rate: {health_data.get("error_rate", 0):.1%}

Concerns:
{concern_list}

Please investigate system performance and take corrective action if needed.

Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    def _format_risk_factors(self, risk_factors) -> str:
        """Format risk factors for display"""
        if not risk_factors:
            return "None specified"
        
        return "\n".join([f"- {rf.factor_type}: {rf.factor_value}" for rf in risk_factors[:5]])
    
    def _setup_logging(self) -> logging.Logger:
        """Setup notification service logging"""
        logger = logging.getLogger("aml_system.notification_service")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger


class NotificationChannel:
    """Base class for notification channels"""
    
    async def send_notification(self, subject: str, message: str, data: Dict[str, Any]):
        """Send notification through this channel"""
        raise NotImplementedError
    
    def is_configured(self) -> bool:
        """Check if channel is properly configured"""
        return True


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    async def send_notification(self, subject: str, message: str, data: Dict[str, Any]):
        """Send email notification"""
        # Mock implementation - would integrate with actual email service
        print(f"EMAIL: {subject}\n{message}")


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel"""
    
    async def send_notification(self, subject: str, message: str, data: Dict[str, Any]):
        """Send Slack notification"""
        # Mock implementation - would integrate with Slack API
        print(f"SLACK: {subject}\n{message}")


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel"""
    
    async def send_notification(self, subject: str, message: str, data: Dict[str, Any]):
        """Send webhook notification"""
        # Mock implementation - would make HTTP POST to webhook URL
        print(f"WEBHOOK: {subject}\n{data}")


class SMSNotificationChannel(NotificationChannel):
    """SMS notification channel"""
    
    async def send_notification(self, subject: str, message: str, data: Dict[str, Any]):
        """Send SMS notification"""
        # Mock implementation - would integrate with SMS service
        print(f"SMS: {subject} - Check AML system for details")