"""
Advanced Metrics Collection System

Comprehensive metrics collection with real-time monitoring,
custom metrics, and intelligent aggregation.
"""
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-wide metrics snapshot"""
    timestamp: datetime
    
    # Performance Metrics
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_bytes: Dict[str, int]
    
    # Application Metrics
    active_connections: int
    request_rate_per_second: float
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    error_rate_percent: float
    
    # AML-Specific Metrics
    transactions_processed_total: int
    transactions_processing_rate: float
    high_risk_transactions_count: int
    sar_generated_count: int
    false_positive_rate: float
    detection_accuracy: float
    
    # Queue Metrics
    pending_analyses: int
    queue_depth: int
    processing_backlog_minutes: float
    
    # Model Performance
    model_inference_time_ms: float
    model_accuracy: float
    feature_extraction_time_ms: float
    
    # External Dependencies
    llm_api_latency_ms: float
    llm_api_error_rate: float
    database_connection_pool_usage: float
    cache_hit_rate: float


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    severity: str  # 'critical', 'warning', 'info'
    metric_name: str
    current_value: float
    threshold_value: float
    description: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.metrics_store = defaultdict(lambda: deque(maxlen=10000))
        self.custom_metrics = {}
        self.alert_rules = {}
        self.active_alerts = {}
        self.collection_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Performance tracking
        self.request_timers = defaultdict(list)
        self.counter_metrics = defaultdict(int)
        self.gauge_metrics = defaultdict(float)
        
    def _default_config(self) -> Dict:
        """Default metrics configuration"""
        return {
            'collection_interval_seconds': 15,
            'retention_hours': 168,  # 7 days
            'batch_size': 100,
            'enable_system_metrics': True,
            'enable_custom_metrics': True,
            'metric_aggregation_windows': [60, 300, 3600],  # 1min, 5min, 1hour
            'alert_cooldown_minutes': 5,
            'export_format': 'prometheus'
        }
    
    def start_collection(self):
        """Start metrics collection"""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect system metrics
                if self.config['enable_system_metrics']:
                    system_metrics = self._collect_system_metrics()
                    self._store_system_metrics(system_metrics)
                
                # Process custom metrics
                if self.config['enable_custom_metrics']:
                    self._process_custom_metrics()
                
                # Check alert conditions
                self._check_alert_rules()
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                time.sleep(self.config['collection_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}")
                time.sleep(1)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # In production, would use psutil, boto3 CloudWatch, etc.
        # For demo, using mock values with some realistic variance
        
        import random
        
        base_time = time.time()
        
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            
            # Performance (simulated)
            cpu_usage_percent=random.uniform(20, 80),
            memory_usage_percent=random.uniform(40, 85),
            disk_usage_percent=random.uniform(30, 70),
            network_io_bytes={'rx': random.randint(1000000, 5000000), 'tx': random.randint(500000, 2000000)},
            
            # Application
            active_connections=random.randint(100, 500),
            request_rate_per_second=random.uniform(50, 200),
            response_time_p50=random.uniform(80, 150),
            response_time_p95=random.uniform(200, 400),
            response_time_p99=random.uniform(500, 800),
            error_rate_percent=random.uniform(0.1, 2.0),
            
            # AML Specific
            transactions_processed_total=self.counter_metrics.get('transactions_total', 0),
            transactions_processing_rate=random.uniform(80, 120),
            high_risk_transactions_count=random.randint(5, 25),
            sar_generated_count=random.randint(1, 8),
            false_positive_rate=random.uniform(1.5, 3.5),
            detection_accuracy=random.uniform(95, 99),
            
            # Queue
            pending_analyses=random.randint(10, 100),
            queue_depth=random.randint(5, 50),
            processing_backlog_minutes=random.uniform(0.5, 5.0),
            
            # Model Performance
            model_inference_time_ms=random.uniform(50, 200),
            model_accuracy=random.uniform(94, 98),
            feature_extraction_time_ms=random.uniform(20, 80),
            
            # External Dependencies
            llm_api_latency_ms=random.uniform(200, 800),
            llm_api_error_rate=random.uniform(0.1, 1.0),
            database_connection_pool_usage=random.uniform(30, 80),
            cache_hit_rate=random.uniform(85, 95)
        )
    
    def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in time series"""
        
        with self.lock:
            # Store individual metrics as time series
            for field_name, value in metrics.__dict__.items():
                if field_name == 'timestamp':
                    continue
                
                if isinstance(value, (int, float)):
                    metric_point = MetricPoint(
                        timestamp=metrics.timestamp,
                        value=float(value),
                        tags={'source': 'system', 'metric_type': field_name}
                    )
                    self.metrics_store[field_name].append(metric_point)
                
                elif isinstance(value, dict):
                    # Handle nested metrics like network_io_bytes
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            metric_name = f"{field_name}_{sub_key}"
                            metric_point = MetricPoint(
                                timestamp=metrics.timestamp,
                                value=float(sub_value),
                                tags={'source': 'system', 'metric_type': metric_name, 'subtype': sub_key}
                            )
                            self.metrics_store[metric_name].append(metric_point)
    
    def record_custom_metric(
        self, 
        metric_name: str, 
        value: float, 
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a custom metric"""
        
        tags = tags or {}
        
        with self.lock:
            metric_point = MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                tags={**tags, 'source': 'custom', 'type': metric_type.value},
                metadata={'metric_type': metric_type}
            )
            
            self.metrics_store[metric_name].append(metric_point)
            
            # Update in-memory counters/gauges for quick access
            if metric_type == MetricType.COUNTER:
                self.counter_metrics[metric_name] += value
            elif metric_type == MetricType.GAUGE:
                self.gauge_metrics[metric_name] = value
    
    def record_timer(self, operation_name: str) -> 'TimerContext':
        """Record operation timing"""
        return TimerContext(self, operation_name)
    
    def increment_counter(self, counter_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self.record_custom_metric(counter_name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, gauge_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        self.record_custom_metric(gauge_name, value, MetricType.GAUGE, tags)
    
    def get_metric_history(
        self, 
        metric_name: str, 
        time_window_minutes: int = 60
    ) -> List[MetricPoint]:
        """Get metric history for specified time window"""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        with self.lock:
            if metric_name not in self.metrics_store:
                return []
            
            return [
                point for point in self.metrics_store[metric_name]
                if point.timestamp >= cutoff_time
            ]
    
    def get_metric_statistics(
        self, 
        metric_name: str, 
        time_window_minutes: int = 60
    ) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        
        history = self.get_metric_history(metric_name, time_window_minutes)
        
        if not history:
            return {}
        
        values = [point.value for point in history]
        
        import statistics
        
        stats = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'latest': values[-1],
            'first': values[0]
        }
        
        # Add percentiles
        if len(values) >= 2:
            sorted_values = sorted(values)
            stats['p95'] = sorted_values[int(0.95 * len(sorted_values))]
            stats['p99'] = sorted_values[int(0.99 * len(sorted_values))]
        
        return stats
    
    def add_alert_rule(
        self, 
        rule_name: str, 
        metric_name: str, 
        condition: str, 
        threshold: float,
        severity: str = 'warning'
    ):
        """Add an alert rule"""
        
        self.alert_rules[rule_name] = {
            'metric_name': metric_name,
            'condition': condition,  # 'gt', 'lt', 'eq'
            'threshold': threshold,
            'severity': severity,
            'last_triggered': None,
            'cooldown_minutes': self.config['alert_cooldown_minutes']
        }
        
        logger.info(f"Added alert rule: {rule_name}")
    
    def _check_alert_rules(self):
        """Check all alert rules against current metrics"""
        
        for rule_name, rule in self.alert_rules.items():
            try:
                current_value = self._get_latest_metric_value(rule['metric_name'])
                if current_value is None:
                    continue
                
                # Check condition
                should_alert = False
                condition = rule['condition']
                threshold = rule['threshold']
                
                if condition == 'gt' and current_value > threshold:
                    should_alert = True
                elif condition == 'lt' and current_value < threshold:
                    should_alert = True
                elif condition == 'eq' and abs(current_value - threshold) < 0.01:
                    should_alert = True
                
                # Check cooldown
                if should_alert:
                    last_triggered = rule.get('last_triggered')
                    if last_triggered:
                        cooldown_period = timedelta(minutes=rule['cooldown_minutes'])
                        if datetime.utcnow() - last_triggered < cooldown_period:
                            continue
                    
                    # Trigger alert
                    self._trigger_alert(rule_name, rule, current_value)
                    rule['last_triggered'] = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {str(e)}")
    
    def _get_latest_metric_value(self, metric_name: str) -> Optional[float]:
        """Get the latest value for a metric"""
        
        with self.lock:
            if metric_name not in self.metrics_store:
                return None
            
            if not self.metrics_store[metric_name]:
                return None
            
            return self.metrics_store[metric_name][-1].value
    
    def _trigger_alert(self, rule_name: str, rule: Dict, current_value: float):
        """Trigger an alert"""
        
        alert = Alert(
            alert_id=f"{rule_name}_{int(time.time())}",
            severity=rule['severity'],
            metric_name=rule['metric_name'],
            current_value=current_value,
            threshold_value=rule['threshold'],
            description=f"Alert: {rule['metric_name']} is {current_value} (threshold: {rule['threshold']})",
            timestamp=datetime.utcnow(),
            tags={'rule': rule_name}
        )
        
        self.active_alerts[alert.alert_id] = alert
        
        logger.warning(f"ALERT TRIGGERED: {alert.description}")
        
        # In production, would send to alert manager, email, Slack, etc.
        self._send_alert_notification(alert)
    
    def _send_alert_notification(self, alert: Alert):
        """Send alert notification (mock implementation)"""
        # In production, would integrate with:
        # - Email/SMS providers
        # - Slack/Teams webhooks
        # - PagerDuty/OpsGenie
        # - Custom notification systems
        pass
    
    def _process_custom_metrics(self):
        """Process and aggregate custom metrics"""
        # Placeholder for custom metric processing logic
        # Could include:
        # - Metric aggregation
        # - Derived metric calculation
        # - Custom alerting logic
        pass
    
    def _cleanup_old_metrics(self):
        """Clean up old metric data"""
        
        retention_cutoff = datetime.utcnow() - timedelta(hours=self.config['retention_hours'])
        
        with self.lock:
            for metric_name, points in self.metrics_store.items():
                # Remove old points
                while points and points[0].timestamp < retention_cutoff:
                    points.popleft()
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """Export metrics in specified format"""
        
        if format_type == 'json':
            return self._export_json()
        elif format_type == 'prometheus':
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON"""
        
        export_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {},
            'alerts': []
        }
        
        # Export recent metrics
        with self.lock:
            for metric_name, points in self.metrics_store.items():
                if points:
                    latest_point = points[-1]
                    export_data['metrics'][metric_name] = {
                        'value': latest_point.value,
                        'timestamp': latest_point.timestamp.isoformat(),
                        'tags': latest_point.tags
                    }
        
        # Export active alerts
        for alert in self.active_alerts.values():
            if not alert.resolved:
                export_data['alerts'].append({
                    'id': alert.alert_id,
                    'severity': alert.severity,
                    'metric': alert.metric_name,
                    'description': alert.description,
                    'timestamp': alert.timestamp.isoformat()
                })
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        
        lines = []
        lines.append("# HELP aml_system_metrics AML System Metrics")
        lines.append("# TYPE aml_system_metrics gauge")
        
        with self.lock:
            for metric_name, points in self.metrics_store.items():
                if points:
                    latest_point = points[-1]
                    
                    # Convert metric name to Prometheus format
                    prom_name = f"aml_{metric_name.replace('-', '_').replace('.', '_')}"
                    
                    # Add tags
                    tag_strings = []
                    for key, value in latest_point.tags.items():
                        tag_strings.append(f'{key}="{value}"')
                    
                    tag_part = f"{{{','.join(tag_strings)}}}" if tag_strings else ""
                    
                    lines.append(f"{prom_name}{tag_part} {latest_point.value}")
        
        return '\n'.join(lines)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        
        dashboard_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_health': self._calculate_system_health(),
            'key_metrics': self._get_key_metrics(),
            'recent_alerts': self._get_recent_alerts(),
            'performance_summary': self._get_performance_summary()
        }
        
        return dashboard_data
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health"""
        
        critical_alerts = sum(
            1 for alert in self.active_alerts.values()
            if not alert.resolved and alert.severity == 'critical'
        )
        
        warning_alerts = sum(
            1 for alert in self.active_alerts.values()
            if not alert.resolved and alert.severity == 'warning'
        )
        
        if critical_alerts > 0:
            return 'critical'
        elif warning_alerts > 2:
            return 'warning'
        else:
            return 'healthy'
    
    def _get_key_metrics(self) -> Dict[str, float]:
        """Get key system metrics"""
        
        key_metrics = {}
        important_metrics = [
            'transactions_processing_rate',
            'response_time_p95',
            'error_rate_percent',
            'detection_accuracy',
            'false_positive_rate'
        ]
        
        for metric_name in important_metrics:
            value = self._get_latest_metric_value(metric_name)
            if value is not None:
                key_metrics[metric_name] = value
        
        return key_metrics
    
    def _get_recent_alerts(self) -> List[Dict]:
        """Get recent alerts"""
        
        recent_alerts = []
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for alert in self.active_alerts.values():
            if alert.timestamp >= cutoff_time:
                recent_alerts.append({
                    'id': alert.alert_id,
                    'severity': alert.severity,
                    'metric': alert.metric_name,
                    'description': alert.description,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved
                })
        
        return sorted(recent_alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        summary = {}
        
        # Transaction processing stats
        txn_rate = self._get_latest_metric_value('transactions_processing_rate')
        if txn_rate:
            summary['transaction_throughput'] = txn_rate
        
        # Response time stats
        response_stats = self.get_metric_statistics('response_time_p95', 60)
        if response_stats:
            summary['response_time_trend'] = {
                'current': response_stats.get('latest', 0),
                'average': response_stats.get('mean', 0),
                'peak': response_stats.get('max', 0)
            }
        
        # Error rate trend
        error_stats = self.get_metric_statistics('error_rate_percent', 60)
        if error_stats:
            summary['error_rate_trend'] = {
                'current': error_stats.get('latest', 0),
                'average': error_stats.get('mean', 0)
            }
        
        return summary


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, operation_name: str):
        self.collector = collector
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.record_custom_metric(
                f"{self.operation_name}_duration_ms",
                duration_ms,
                MetricType.TIMER,
                tags={'operation': self.operation_name}
            )


def create_sample_metrics() -> SystemMetrics:
    """Create sample system metrics for demonstration"""
    
    return SystemMetrics(
        timestamp=datetime.utcnow(),
        cpu_usage_percent=45.2,
        memory_usage_percent=68.1,
        disk_usage_percent=34.7,
        network_io_bytes={'rx': 2500000, 'tx': 1200000},
        active_connections=250,
        request_rate_per_second=125.4,
        response_time_p50=95.2,
        response_time_p95=245.8,
        response_time_p99=567.3,
        error_rate_percent=0.8,
        transactions_processed_total=15420,
        transactions_processing_rate=95.7,
        high_risk_transactions_count=12,
        sar_generated_count=3,
        false_positive_rate=2.1,
        detection_accuracy=96.8,
        pending_analyses=25,
        queue_depth=18,
        processing_backlog_minutes=1.8,
        model_inference_time_ms=124.5,
        model_accuracy=96.2,
        feature_extraction_time_ms=67.3,
        llm_api_latency_ms=456.7,
        llm_api_error_rate=0.3,
        database_connection_pool_usage=45.2,
        cache_hit_rate=92.1
    )