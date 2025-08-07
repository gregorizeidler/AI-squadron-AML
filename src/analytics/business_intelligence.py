"""
Business Intelligence Engine for AML Operations

Advanced analytics and business intelligence capabilities
for strategic decision making and operational insights.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from decimal import Decimal
from collections import defaultdict

from ..models.analysis import AMLAnalysisResult
from ..models.enums import RiskLevel, DecisionType

logger = logging.getLogger(__name__)


@dataclass
class KPIMetric:
    """Key Performance Indicator metric"""
    name: str
    value: float
    target: float
    trend: str  # 'up', 'down', 'stable'
    variance_percentage: float
    description: str
    category: str  # 'operational', 'compliance', 'financial'


@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric_name: str
    period: str
    trend_direction: str
    growth_rate: float
    seasonality_detected: bool
    forecast_next_period: float
    confidence_interval: Tuple[float, float]


@dataclass
class RiskInsight:
    """Risk-related business insight"""
    insight_type: str
    description: str
    impact_level: str  # 'high', 'medium', 'low'
    recommended_action: str
    estimated_cost_benefit: Optional[float]
    timeline: str


class BusinessIntelligenceEngine:
    """Advanced business intelligence for AML operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.kpi_thresholds = self._initialize_kpi_thresholds()
        
    def _default_config(self) -> Dict:
        """Default BI configuration"""
        return {
            'lookback_days': 90,
            'trend_periods': [7, 30, 90, 365],
            'seasonality_threshold': 0.1,
            'outlier_threshold': 2.0,
            'min_data_points': 30,
            'forecast_horizon': 30
        }
    
    def _initialize_kpi_thresholds(self) -> Dict[str, Dict]:
        """Initialize KPI thresholds and targets"""
        return {
            'false_positive_rate': {'target': 0.02, 'warning': 0.05, 'critical': 0.10},
            'detection_rate': {'target': 0.95, 'warning': 0.90, 'critical': 0.85},
            'investigation_time': {'target': 2.0, 'warning': 4.0, 'critical': 8.0},  # hours
            'sar_filing_rate': {'target': 0.15, 'warning': 0.25, 'critical': 0.35},
            'system_uptime': {'target': 0.999, 'warning': 0.995, 'critical': 0.99},
            'processing_latency': {'target': 200, 'warning': 500, 'critical': 1000},  # ms
            'analyst_productivity': {'target': 20, 'warning': 15, 'critical': 10}  # cases/day
        }
    
    def generate_executive_dashboard(
        self, 
        analysis_results: List[AMLAnalysisResult],
        time_period: int = 30
    ) -> Dict[str, Any]:
        """Generate executive-level dashboard data"""
        
        # Filter recent results
        cutoff_date = datetime.utcnow() - timedelta(days=time_period)
        recent_results = [
            r for r in analysis_results 
            if r.analysis_timestamp >= cutoff_date
        ]
        
        # Calculate executive KPIs
        executive_kpis = self._calculate_executive_kpis(recent_results)
        
        # Risk distribution analysis
        risk_distribution = self._analyze_risk_distribution(recent_results)
        
        # Financial impact analysis
        financial_impact = self._calculate_financial_impact(recent_results)
        
        # Compliance metrics
        compliance_metrics = self._calculate_compliance_metrics(recent_results)
        
        # Strategic insights
        strategic_insights = self._generate_strategic_insights(
            recent_results, executive_kpis
        )
        
        return {
            'period': f"Last {time_period} days",
            'generated_at': datetime.utcnow().isoformat(),
            'executive_kpis': executive_kpis,
            'risk_distribution': risk_distribution,
            'financial_impact': financial_impact,
            'compliance_metrics': compliance_metrics,
            'strategic_insights': strategic_insights,
            'summary': self._generate_executive_summary(
                executive_kpis, strategic_insights
            )
        }
    
    def _calculate_executive_kpis(
        self, 
        results: List[AMLAnalysisResult]
    ) -> List[KPIMetric]:
        """Calculate executive-level KPIs"""
        
        kpis = []
        
        if not results:
            return kpis
        
        total_cases = len(results)
        
        # Detection Rate
        high_risk_cases = sum(
            1 for r in results 
            if r.risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        )
        detection_rate = high_risk_cases / total_cases if total_cases > 0 else 0
        
        kpis.append(KPIMetric(
            name="Detection Rate",
            value=detection_rate,
            target=self.kpi_thresholds['detection_rate']['target'],
            trend=self._calculate_trend(detection_rate, 0.93),  # Previous period mock
            variance_percentage=((detection_rate - 0.93) / 0.93) * 100,
            description="Percentage of high-risk cases detected",
            category="operational"
        ))
        
        # SAR Filing Rate
        sar_cases = sum(1 for r in results if r.sar_recommended)
        sar_rate = sar_cases / total_cases if total_cases > 0 else 0
        
        kpis.append(KPIMetric(
            name="SAR Filing Rate",
            value=sar_rate,
            target=self.kpi_thresholds['sar_filing_rate']['target'],
            trend=self._calculate_trend(sar_rate, 0.18),
            variance_percentage=((sar_rate - 0.18) / 0.18) * 100,
            description="Percentage of cases requiring SAR filing",
            category="compliance"
        ))
        
        # False Positive Rate (estimated)
        approved_cases = sum(1 for r in results if r.transaction_approved)
        high_risk_approved = sum(
            1 for r in results 
            if r.transaction_approved and r.risk_assessment.risk_score > 75
        )
        false_positive_rate = high_risk_approved / max(high_risk_cases, 1)
        
        kpis.append(KPIMetric(
            name="False Positive Rate",
            value=false_positive_rate,
            target=self.kpi_thresholds['false_positive_rate']['target'],
            trend=self._calculate_trend(false_positive_rate, 0.025),
            variance_percentage=((false_positive_rate - 0.025) / 0.025) * 100,
            description="Percentage of false positive alerts",
            category="operational"
        ))
        
        # Average Risk Score
        avg_risk_score = np.mean([r.risk_assessment.risk_score for r in results])
        
        kpis.append(KPIMetric(
            name="Average Risk Score",
            value=avg_risk_score,
            target=45.0,  # Target average
            trend=self._calculate_trend(avg_risk_score, 42.0),
            variance_percentage=((avg_risk_score - 42.0) / 42.0) * 100,
            description="Average risk score across all transactions",
            category="operational"
        ))
        
        return kpis
    
    def _calculate_trend(self, current: float, previous: float) -> str:
        """Calculate trend direction"""
        if abs(current - previous) / previous < 0.05:  # 5% threshold
            return "stable"
        elif current > previous:
            return "up"
        else:
            return "down"
    
    def _analyze_risk_distribution(
        self, 
        results: List[AMLAnalysisResult]
    ) -> Dict[str, Any]:
        """Analyze risk distribution across different dimensions"""
        
        if not results:
            return {}
        
        # Risk level distribution
        risk_levels = defaultdict(int)
        for result in results:
            risk_levels[result.risk_assessment.risk_level.value] += 1
        
        total = len(results)
        risk_distribution = {
            level: count / total for level, count in risk_levels.items()
        }
        
        # Risk score histogram
        risk_scores = [r.risk_assessment.risk_score for r in results]
        score_bins = np.histogram(risk_scores, bins=10, range=(0, 100))
        
        # Geographic risk distribution
        geo_risks = defaultdict(list)
        for result in results:
            # Mock geographic data - would extract from transaction details
            country = "US"  # Would extract from actual transaction
            geo_risks[country].append(result.risk_assessment.risk_score)
        
        geo_distribution = {
            country: {
                'count': len(scores),
                'avg_risk': np.mean(scores),
                'max_risk': max(scores)
            }
            for country, scores in geo_risks.items()
        }
        
        return {
            'risk_level_distribution': risk_distribution,
            'risk_score_histogram': {
                'bins': score_bins[1].tolist(),
                'counts': score_bins[0].tolist()
            },
            'geographic_distribution': geo_distribution,
            'statistics': {
                'mean_risk_score': np.mean(risk_scores),
                'median_risk_score': np.median(risk_scores),
                'std_risk_score': np.std(risk_scores),
                'max_risk_score': max(risk_scores),
                'min_risk_score': min(risk_scores)
            }
        }
    
    def _calculate_financial_impact(
        self, 
        results: List[AMLAnalysisResult]
    ) -> Dict[str, Any]:
        """Calculate financial impact of AML operations"""
        
        if not results:
            return {}
        
        # Cost calculations (estimated)
        investigation_hours = len(results) * 2.5  # Average hours per investigation
        cost_per_hour = 75  # USD per hour
        investigation_cost = investigation_hours * cost_per_hour
        
        # False positive costs
        false_positives = sum(
            1 for r in results 
            if r.risk_assessment.risk_score > 50 and r.transaction_approved
        )
        false_positive_cost = false_positives * 150  # Cost per false positive
        
        # Compliance costs
        sar_count = sum(1 for r in results if r.sar_recommended)
        sar_cost = sar_count * 500  # Cost per SAR filing
        
        # Potential savings (prevented laundering)
        high_risk_blocked = sum(
            1 for r in results 
            if not r.transaction_approved and r.risk_assessment.risk_level == RiskLevel.CRITICAL
        )
        prevented_laundering = high_risk_blocked * 250000  # Estimated prevented amount
        
        total_cost = investigation_cost + false_positive_cost + sar_cost
        net_benefit = prevented_laundering - total_cost
        roi = (net_benefit / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            'total_operational_cost': total_cost,
            'investigation_cost': investigation_cost,
            'false_positive_cost': false_positive_cost,
            'compliance_cost': sar_cost,
            'prevented_money_laundering': prevented_laundering,
            'net_financial_benefit': net_benefit,
            'roi_percentage': roi,
            'cost_per_transaction': total_cost / len(results),
            'cost_breakdown': {
                'investigations': investigation_cost / total_cost,
                'false_positives': false_positive_cost / total_cost,
                'compliance': sar_cost / total_cost
            }
        }
    
    def _calculate_compliance_metrics(
        self, 
        results: List[AMLAnalysisResult]
    ) -> Dict[str, Any]:
        """Calculate compliance-related metrics"""
        
        if not results:
            return {}
        
        # Regulatory compliance
        total_cases = len(results)
        sar_cases = sum(1 for r in results if r.sar_recommended)
        
        # Time to detection (mock - would calculate from actual timestamps)
        avg_detection_time = 2.5  # hours
        
        # Coverage metrics
        high_value_transactions = sum(
            1 for r in results 
            if r.risk_assessment.risk_score > 75
        )
        
        coverage_rate = high_value_transactions / total_cases
        
        # Audit readiness
        documented_cases = sum(
            1 for r in results 
            if len(r.decision_path) > 0
        )
        
        audit_readiness = documented_cases / total_cases
        
        return {
            'sar_filing_rate': sar_cases / total_cases,
            'average_detection_time_hours': avg_detection_time,
            'high_risk_coverage_rate': coverage_rate,
            'audit_readiness_score': audit_readiness,
            'regulatory_metrics': {
                'cases_requiring_sar': sar_cases,
                'cases_requiring_edd': sum(
                    1 for r in results if r.requires_human_review
                ),
                'blocked_transactions': sum(
                    1 for r in results if not r.transaction_approved
                )
            },
            'compliance_score': (
                (1 - abs(sar_cases / total_cases - 0.15)) * 0.4 +  # Target SAR rate 15%
                min(audit_readiness, 1.0) * 0.6
            )
        }
    
    def _generate_strategic_insights(
        self, 
        results: List[AMLAnalysisResult],
        kpis: List[KPIMetric]
    ) -> List[RiskInsight]:
        """Generate strategic business insights"""
        
        insights = []
        
        if not results:
            return insights
        
        # Insight 1: False Positive Optimization
        fp_kpi = next((k for k in kpis if k.name == "False Positive Rate"), None)
        if fp_kpi and fp_kpi.value > fp_kpi.target:
            insights.append(RiskInsight(
                insight_type="operational_efficiency",
                description=f"False positive rate of {fp_kpi.value:.1%} exceeds target of {fp_kpi.target:.1%}",
                impact_level="high",
                recommended_action="Implement machine learning model tuning and analyst feedback loops",
                estimated_cost_benefit=500000,  # Annual savings
                timeline="3 months"
            ))
        
        # Insight 2: Detection Rate Analysis
        detection_kpi = next((k for k in kpis if k.name == "Detection Rate"), None)
        if detection_kpi and detection_kpi.value < detection_kpi.target:
            insights.append(RiskInsight(
                insight_type="risk_coverage",
                description=f"Detection rate of {detection_kpi.value:.1%} is below target",
                impact_level="critical",
                recommended_action="Enhance graph neural network models and add new risk indicators",
                estimated_cost_benefit=1000000,  # Prevented losses
                timeline="6 months"
            ))
        
        # Insight 3: Resource Optimization
        total_cases = len(results)
        manual_review_cases = sum(1 for r in results if r.requires_human_review)
        manual_rate = manual_review_cases / total_cases
        
        if manual_rate > 0.3:  # > 30% manual review
            insights.append(RiskInsight(
                insight_type="resource_optimization",
                description=f"Manual review rate of {manual_rate:.1%} indicates automation opportunity",
                impact_level="medium",
                recommended_action="Implement automated case prioritization and smart routing",
                estimated_cost_benefit=300000,  # Cost savings
                timeline="4 months"
            ))
        
        # Insight 4: Technology Enhancement
        avg_risk = np.mean([r.risk_assessment.risk_score for r in results])
        if avg_risk < 40:  # Low average risk might indicate missed threats
            insights.append(RiskInsight(
                insight_type="technology_enhancement",
                description="Low average risk scores may indicate gaps in threat detection",
                impact_level="high",
                recommended_action="Deploy advanced feature engineering and community detection algorithms",
                estimated_cost_benefit=750000,  # Enhanced detection value
                timeline="5 months"
            ))
        
        return insights
    
    def _generate_executive_summary(
        self, 
        kpis: List[KPIMetric], 
        insights: List[RiskInsight]
    ) -> str:
        """Generate executive summary"""
        
        # Key metrics summary
        detection_rate = next(
            (k.value for k in kpis if k.name == "Detection Rate"), 0
        )
        sar_rate = next(
            (k.value for k in kpis if k.name == "SAR Filing Rate"), 0
        )
        fp_rate = next(
            (k.value for k in kpis if k.name == "False Positive Rate"), 0
        )
        
        # Trend analysis
        improving_metrics = sum(1 for k in kpis if k.trend == "up" and "Rate" in k.name)
        declining_metrics = sum(1 for k in kpis if k.trend == "down" and "Rate" in k.name)
        
        # Priority insights
        critical_insights = [i for i in insights if i.impact_level == "critical"]
        high_insights = [i for i in insights if i.impact_level == "high"]
        
        summary = f"""
        AML Operations Executive Summary:
        
        Key Performance Indicators:
        • Detection Rate: {detection_rate:.1%}
        • SAR Filing Rate: {sar_rate:.1%}  
        • False Positive Rate: {fp_rate:.1%}
        
        Operational Status:
        • {improving_metrics} metrics showing improvement
        • {declining_metrics} metrics requiring attention
        • {len(critical_insights)} critical insights identified
        • {len(high_insights)} high-impact opportunities available
        
        Strategic Priorities:
        {chr(10).join([f"• {i.recommended_action}" for i in critical_insights + high_insights[:2]])}
        
        Financial Impact:
        • Estimated annual benefit from recommended improvements: ${sum(i.estimated_cost_benefit or 0 for i in insights):,.0f}
        """
        
        return summary.strip()
    
    def generate_trend_analysis(
        self, 
        historical_data: List[Dict], 
        metric_name: str,
        period_days: int = 30
    ) -> TrendAnalysis:
        """Generate trend analysis for specific metrics"""
        
        if len(historical_data) < self.config['min_data_points']:
            return TrendAnalysis(
                metric_name=metric_name,
                period=f"{period_days} days",
                trend_direction="insufficient_data",
                growth_rate=0.0,
                seasonality_detected=False,
                forecast_next_period=0.0,
                confidence_interval=(0.0, 0.0)
            )
        
        # Extract values and timestamps
        values = [d.get(metric_name, 0) for d in historical_data]
        timestamps = [d.get('timestamp', datetime.utcnow()) for d in historical_data]
        
        # Calculate growth rate
        if len(values) >= 2:
            recent_avg = np.mean(values[-7:])  # Last week
            previous_avg = np.mean(values[-14:-7])  # Previous week
            growth_rate = ((recent_avg - previous_avg) / previous_avg) * 100 if previous_avg != 0 else 0
        else:
            growth_rate = 0
        
        # Determine trend direction
        if abs(growth_rate) < 5:
            trend_direction = "stable"
        elif growth_rate > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # Seasonality detection (simplified)
        seasonality_detected = self._detect_seasonality(values)
        
        # Simple forecast
        forecast_next_period = values[-1] * (1 + growth_rate / 100) if values else 0
        
        # Confidence interval (simplified)
        std_dev = np.std(values) if values else 0
        confidence_interval = (
            forecast_next_period - 1.96 * std_dev,
            forecast_next_period + 1.96 * std_dev
        )
        
        return TrendAnalysis(
            metric_name=metric_name,
            period=f"{period_days} days",
            trend_direction=trend_direction,
            growth_rate=growth_rate,
            seasonality_detected=seasonality_detected,
            forecast_next_period=forecast_next_period,
            confidence_interval=confidence_interval
        )
    
    def _detect_seasonality(self, values: List[float]) -> bool:
        """Simple seasonality detection"""
        if len(values) < 14:  # Need at least 2 weeks
            return False
        
        # Check for weekly patterns
        weekly_pattern = []
        for i in range(7):
            week_values = values[i::7]  # Every 7th value starting from i
            if len(week_values) >= 2:
                weekly_pattern.append(np.mean(week_values))
        
        if len(weekly_pattern) >= 7:
            # If standard deviation is significant relative to mean, there's seasonality
            pattern_std = np.std(weekly_pattern)
            pattern_mean = np.mean(weekly_pattern)
            if pattern_mean > 0:
                coefficient_of_variation = pattern_std / pattern_mean
                return coefficient_of_variation > self.config['seasonality_threshold']
        
        return False


def create_sample_bi_dashboard() -> Dict[str, Any]:
    """Create sample BI dashboard for demonstration"""
    
    sample_kpis = [
        KPIMetric(
            name="Detection Rate",
            value=0.952,
            target=0.95,
            trend="up",
            variance_percentage=2.3,
            description="Percentage of high-risk cases detected",
            category="operational"
        ),
        KPIMetric(
            name="False Positive Rate", 
            value=0.018,
            target=0.02,
            trend="down",
            variance_percentage=-10.0,
            description="Percentage of false positive alerts",
            category="operational"
        )
    ]
    
    sample_insights = [
        RiskInsight(
            insight_type="operational_efficiency",
            description="Machine learning model optimization reduced false positives by 15%",
            impact_level="high",
            recommended_action="Expand ML optimization to additional transaction types",
            estimated_cost_benefit=750000,
            timeline="3 months"
        )
    ]
    
    return {
        'period': "Last 30 days",
        'generated_at': datetime.utcnow().isoformat(),
        'executive_kpis': sample_kpis,
        'risk_distribution': {
            'LOW': 0.65,
            'MEDIUM': 0.25,
            'HIGH': 0.08,
            'CRITICAL': 0.02
        },
        'financial_impact': {
            'total_operational_cost': 245000,
            'prevented_money_laundering': 2500000,
            'net_financial_benefit': 2255000,
            'roi_percentage': 920.4
        },
        'strategic_insights': sample_insights
    }