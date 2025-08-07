"""
Monitoring Service for AML Detection System

This service provides comprehensive monitoring, metrics collection,
and system health tracking for the AML detection system.
"""
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading

from ..models.analysis import AMLAnalysisResult


class MonitoringService:
    """
    Service for monitoring system performance and metrics.
    
    Provides:
    - Real-time performance metrics
    - Error tracking and alerting
    - System health monitoring
    - Analysis result metrics
    - Throughput and latency tracking
    """
    
    def __init__(self):
        """Initialize the monitoring service"""
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Metrics storage
        self.metrics = {
            "analyses_completed": 0,
            "analyses_failed": 0,
            "sar_generated": 0,
            "reviews_required": 0,
            "high_risk_transactions": 0,
            "processing_times": deque(maxlen=1000),  # Last 1000 processing times
            "error_counts": defaultdict(int),
            "risk_score_distribution": defaultdict(int)
        }
        
        # Real-time tracking
        self.hourly_metrics = defaultdict(lambda: defaultdict(int))
        self.daily_metrics = defaultdict(lambda: defaultdict(int))
        
        # Thread safety
        self.lock = threading.Lock()
    
    def start(self):
        """Start the monitoring service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        # Reset metrics
        with self.lock:
            self.metrics["analyses_completed"] = 0
            self.metrics["analyses_failed"] = 0
            self.metrics["sar_generated"] = 0
            self.metrics["reviews_required"] = 0
            self.metrics["high_risk_transactions"] = 0
    
    def stop(self):
        """Stop the monitoring service"""
        self.is_running = False
    
    def record_analysis_completed(self, result: AMLAnalysisResult):
        """Record completion of an analysis"""
        with self.lock:
            self.metrics["analyses_completed"] += 1
            
            # Record processing time
            if result.processing_time_seconds:
                self.metrics["processing_times"].append(result.processing_time_seconds)
            
            # Record SAR generation
            if result.sar_recommended:
                self.metrics["sar_generated"] += 1
            
            # Record review requirements
            if result.requires_human_review:
                self.metrics["reviews_required"] += 1
            
            # Record high-risk transactions
            if result.risk_assessment.risk_score >= 75:
                self.metrics["high_risk_transactions"] += 1
            
            # Record risk score distribution
            risk_bucket = (result.risk_assessment.risk_score // 10) * 10
            self.metrics["risk_score_distribution"][risk_bucket] += 1
            
            # Update hourly and daily metrics
            now = datetime.utcnow()
            hour_key = now.strftime("%Y-%m-%d %H:00")
            day_key = now.strftime("%Y-%m-%d")
            
            self.hourly_metrics[hour_key]["analyses_completed"] += 1
            self.daily_metrics[day_key]["analyses_completed"] += 1
            
            if result.sar_recommended:
                self.hourly_metrics[hour_key]["sar_generated"] += 1
                self.daily_metrics[day_key]["sar_generated"] += 1
    
    def record_analysis_error(self, transaction_id: str, error_message: str):
        """Record an analysis error"""
        with self.lock:
            self.metrics["analyses_failed"] += 1
            
            # Categorize error
            error_category = self._categorize_error(error_message)
            self.metrics["error_counts"][error_category] += 1
            
            # Update hourly and daily metrics
            now = datetime.utcnow()
            hour_key = now.strftime("%Y-%m-%d %H:00")
            day_key = now.strftime("%Y-%m-%d")
            
            self.hourly_metrics[hour_key]["analyses_failed"] += 1
            self.daily_metrics[day_key]["analyses_failed"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        with self.lock:
            uptime = None
            if self.start_time:
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Calculate performance metrics
            processing_times = list(self.metrics["processing_times"])
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            throughput = 0
            if uptime and uptime > 0:
                throughput = self.metrics["analyses_completed"] / (uptime / 3600)  # per hour
            
            return {
                "system": {
                    "is_running": self.is_running,
                    "uptime_seconds": uptime,
                    "start_time": self.start_time.isoformat() if self.start_time else None
                },
                "performance": {
                    "throughput_per_hour": round(throughput, 2),
                    "average_processing_time": round(avg_processing_time, 2),
                    "total_analyses": self.metrics["analyses_completed"],
                    "total_errors": self.metrics["analyses_failed"]
                },
                "aml_metrics": {
                    "sar_generated": self.metrics["sar_generated"],
                    "reviews_required": self.metrics["reviews_required"],
                    "high_risk_transactions": self.metrics["high_risk_transactions"],
                    "risk_score_distribution": dict(self.metrics["risk_score_distribution"])
                },
                "error_analysis": dict(self.metrics["error_counts"]),
                "quality_metrics": self._calculate_quality_metrics()
            }
    
    def get_hourly_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get hourly metrics for the specified number of hours"""
        with self.lock:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            
            recent_metrics = {}
            for hour_key, metrics in self.hourly_metrics.items():
                hour_time = datetime.strptime(hour_key, "%Y-%m-%d %H:00")
                if hour_time >= cutoff:
                    recent_metrics[hour_key] = dict(metrics)
            
            return recent_metrics
    
    def get_daily_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get daily metrics for the specified number of days"""
        with self.lock:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            recent_metrics = {}
            for day_key, metrics in self.daily_metrics.items():
                day_time = datetime.strptime(day_key, "%Y-%m-%d")
                if day_time >= cutoff:
                    recent_metrics[day_key] = dict(metrics)
            
            return recent_metrics
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health indicators"""
        with self.lock:
            total_analyses = self.metrics["analyses_completed"] + self.metrics["analyses_failed"]
            error_rate = 0
            if total_analyses > 0:
                error_rate = self.metrics["analyses_failed"] / total_analyses
            
            # Processing time health
            processing_times = list(self.metrics["processing_times"])
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Determine health status
            health_status = "HEALTHY"
            concerns = []
            
            if error_rate > 0.1:  # More than 10% error rate
                health_status = "DEGRADED"
                concerns.append(f"High error rate: {error_rate:.1%}")
            
            if avg_time > 300:  # More than 5 minutes average processing
                health_status = "SLOW"
                concerns.append(f"Slow processing: {avg_time:.1f}s average")
            
            if not self.is_running:
                health_status = "STOPPED"
                concerns.append("Monitoring service is not running")
            
            return {
                "status": health_status,
                "error_rate": round(error_rate, 4),
                "average_processing_time": round(avg_time, 2),
                "total_analyses": total_analyses,
                "concerns": concerns,
                "last_updated": datetime.utcnow().isoformat()
            }
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error by type"""
        error_lower = error_message.lower()
        
        if "timeout" in error_lower:
            return "TIMEOUT"
        elif "connection" in error_lower or "network" in error_lower:
            return "NETWORK"
        elif "authentication" in error_lower or "unauthorized" in error_lower:
            return "AUTH"
        elif "llm" in error_lower or "model" in error_lower:
            return "LLM"
        elif "validation" in error_lower:
            return "VALIDATION"
        else:
            return "OTHER"
    
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate quality metrics for the AML system"""
        with self.lock:
            total_analyses = self.metrics["analyses_completed"]
            
            if total_analyses == 0:
                return {
                    "sar_rate": 0,
                    "review_rate": 0,
                    "high_risk_rate": 0,
                    "efficiency_score": 0
                }
            
            sar_rate = self.metrics["sar_generated"] / total_analyses
            review_rate = self.metrics["reviews_required"] / total_analyses
            high_risk_rate = self.metrics["high_risk_transactions"] / total_analyses
            
            # Efficiency score (lower is better - fewer reviews needed)
            efficiency_score = 1 - review_rate
            
            return {
                "sar_rate": round(sar_rate, 4),
                "review_rate": round(review_rate, 4),
                "high_risk_rate": round(high_risk_rate, 4),
                "efficiency_score": round(efficiency_score, 4)
            }
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of system performance"""
        metrics = self.get_metrics()
        health = self.get_system_health()
        
        report = f"""
AML System Monitoring Report
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

SYSTEM STATUS: {health['status']}
Uptime: {metrics['system']['uptime_seconds']:.0f} seconds

PERFORMANCE METRICS:
- Total Analyses: {metrics['performance']['total_analyses']}
- Throughput: {metrics['performance']['throughput_per_hour']:.1f} analyses/hour
- Average Processing Time: {metrics['performance']['average_processing_time']:.1f} seconds
- Error Rate: {health['error_rate']:.1%}

AML METRICS:
- SARs Generated: {metrics['aml_metrics']['sar_generated']}
- Reviews Required: {metrics['aml_metrics']['reviews_required']}
- High-Risk Transactions: {metrics['aml_metrics']['high_risk_transactions']}

QUALITY METRICS:
- SAR Rate: {metrics['quality_metrics']['sar_rate']:.1%}
- Review Rate: {metrics['quality_metrics']['review_rate']:.1%}
- Efficiency Score: {metrics['quality_metrics']['efficiency_score']:.1%}
"""
        
        if health['concerns']:
            report += f"\nCONCERNS:\n"
            for concern in health['concerns']:
                report += f"- {concern}\n"
        
        return report