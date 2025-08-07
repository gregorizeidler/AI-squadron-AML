"""
Advanced Feature Engineering for AML Detection

Sophisticated feature extraction and engineering techniques
for enhanced money laundering detection capabilities.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from decimal import Decimal
from collections import defaultdict, Counter

from ..models.transaction import Transaction
from ..models.customer import Customer
from ..models.enums import AssetType

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for engineered features"""
    temporal_features: Dict[str, float]
    behavioral_features: Dict[str, float]
    network_features: Dict[str, float]
    statistical_features: Dict[str, float]
    embedding_features: Dict[str, List[float]]
    metadata: Dict[str, Any]


@dataclass
class TemporalPattern:
    """Represents temporal patterns in transactions"""
    pattern_type: str
    pattern_strength: float
    time_window: str
    description: str
    risk_indicator: bool


class AdvancedFeatureEngineer:
    """Advanced feature engineering for AML detection"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.time_windows = ['1d', '7d', '30d', '90d', '365d']
        self.statistical_functions = ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt']
        
    def _default_config(self) -> Dict:
        """Default configuration for feature engineering"""
        return {
            'temporal_windows': [1, 7, 30, 90, 365],  # days
            'min_transactions_for_pattern': 5,
            'velocity_thresholds': [10, 50, 100],  # transactions per day
            'amount_percentiles': [5, 25, 50, 75, 95],
            'seasonality_periods': [7, 30, 365],  # weekly, monthly, yearly
            'embedding_dimensions': 128,
            'behavioral_windows': [3, 7, 14, 30]  # days for behavioral analysis
        }
    
    def extract_comprehensive_features(
        self, 
        target_transaction: Transaction,
        customer: Customer,
        historical_transactions: List[Transaction],
        network_context: Optional[Dict] = None
    ) -> FeatureSet:
        """Extract comprehensive feature set"""
        
        # Temporal features
        temporal_features = self._extract_temporal_features(
            target_transaction, historical_transactions
        )
        
        # Behavioral features
        behavioral_features = self._extract_behavioral_features(
            customer, historical_transactions
        )
        
        # Network features
        network_features = self._extract_network_features(
            target_transaction, customer, network_context or {}
        )
        
        # Statistical features
        statistical_features = self._extract_statistical_features(
            target_transaction, historical_transactions
        )
        
        # Embedding features
        embedding_features = self._extract_embedding_features(
            target_transaction, customer, historical_transactions
        )
        
        # Metadata
        metadata = self._extract_metadata(target_transaction, customer)
        
        return FeatureSet(
            temporal_features=temporal_features,
            behavioral_features=behavioral_features,
            network_features=network_features,
            statistical_features=statistical_features,
            embedding_features=embedding_features,
            metadata=metadata
        )
    
    def _extract_temporal_features(
        self, 
        transaction: Transaction, 
        historical: List[Transaction]
    ) -> Dict[str, float]:
        """Extract temporal pattern features"""
        
        features = {}
        
        # Time-based features
        features['hour_of_day'] = transaction.timestamp.hour
        features['day_of_week'] = transaction.timestamp.weekday()
        features['day_of_month'] = transaction.timestamp.day
        features['month_of_year'] = transaction.timestamp.month
        features['is_weekend'] = float(transaction.timestamp.weekday() >= 5)
        features['is_business_hours'] = float(9 <= transaction.timestamp.hour <= 17)
        
        # Seasonal features
        features['quarter'] = (transaction.timestamp.month - 1) // 3 + 1
        features['day_of_year'] = transaction.timestamp.timetuple().tm_yday
        
        # Time since patterns
        if historical:
            historical_sorted = sorted(historical, key=lambda x: x.timestamp)
            last_transaction = historical_sorted[-1]
            
            time_diff = transaction.timestamp - last_transaction.timestamp
            features['hours_since_last'] = time_diff.total_seconds() / 3600
            features['days_since_last'] = time_diff.days
            
            # Velocity features
            for window_days in self.config['temporal_windows']:
                window_start = transaction.timestamp - timedelta(days=window_days)
                window_transactions = [
                    t for t in historical 
                    if window_start <= t.timestamp <= transaction.timestamp
                ]
                
                features[f'count_{window_days}d'] = len(window_transactions)
                features[f'volume_{window_days}d'] = sum(
                    float(t.amount) for t in window_transactions
                )
                features[f'velocity_{window_days}d'] = len(window_transactions) / max(window_days, 1)
                
                # Time concentration
                if window_transactions:
                    timestamps = [t.timestamp for t in window_transactions]
                    time_span = (max(timestamps) - min(timestamps)).total_seconds()
                    features[f'time_concentration_{window_days}d'] = (
                        len(window_transactions) / max(time_span / 3600, 1)  # transactions per hour
                    )
        
        # Frequency analysis
        features.update(self._extract_frequency_features(transaction, historical))
        
        return features
    
    def _extract_frequency_features(
        self, 
        transaction: Transaction, 
        historical: List[Transaction]
    ) -> Dict[str, float]:
        """Extract frequency-based temporal features"""
        
        features = {}
        
        if len(historical) < 2:
            return features
        
        # Inter-transaction intervals
        historical_sorted = sorted(historical, key=lambda x: x.timestamp)
        intervals = []
        
        for i in range(1, len(historical_sorted)):
            interval = (
                historical_sorted[i].timestamp - historical_sorted[i-1].timestamp
            ).total_seconds() / 3600  # hours
            intervals.append(interval)
        
        if intervals:
            features['avg_interval_hours'] = np.mean(intervals)
            features['std_interval_hours'] = np.std(intervals)
            features['min_interval_hours'] = np.min(intervals)
            features['max_interval_hours'] = np.max(intervals)
            
            # Regularity score (lower standard deviation = more regular)
            if features['avg_interval_hours'] > 0:
                features['regularity_score'] = (
                    features['std_interval_hours'] / features['avg_interval_hours']
                )
            
            # Burst detection
            short_intervals = [i for i in intervals if i < 1]  # < 1 hour
            features['burst_ratio'] = len(short_intervals) / len(intervals)
        
        # Daily pattern analysis
        hourly_counts = defaultdict(int)
        for t in historical:
            hourly_counts[t.timestamp.hour] += 1
        
        if hourly_counts:
            hour_values = list(hourly_counts.values())
            features['hourly_entropy'] = self._calculate_entropy(hour_values)
            features['peak_hour_ratio'] = max(hour_values) / sum(hour_values)
        
        return features
    
    def _extract_behavioral_features(
        self, 
        customer: Customer, 
        historical: List[Transaction]
    ) -> Dict[str, float]:
        """Extract behavioral pattern features"""
        
        features = {}
        
        # Customer profile features
        features['account_age_days'] = customer.account_age_days
        features['total_historical_transactions'] = len(historical)
        features['customer_risk_score'] = customer.risk_score or 0
        
        # Transaction patterns over different windows
        for window_days in self.config['behavioral_windows']:
            cutoff_date = datetime.utcnow() - timedelta(days=window_days)
            recent_transactions = [
                t for t in historical if t.timestamp >= cutoff_date
            ]
            
            if recent_transactions:
                amounts = [float(t.amount) for t in recent_transactions]
                
                features[f'avg_amount_{window_days}d'] = np.mean(amounts)
                features[f'std_amount_{window_days}d'] = np.std(amounts)
                features[f'total_amount_{window_days}d'] = sum(amounts)
                features[f'transaction_count_{window_days}d'] = len(amounts)
                
                # Behavioral change detection
                if window_days > 7:  # Compare with shorter window
                    short_window = window_days // 2
                    short_cutoff = datetime.utcnow() - timedelta(days=short_window)
                    short_transactions = [
                        t for t in recent_transactions if t.timestamp >= short_cutoff
                    ]
                    
                    if short_transactions:
                        short_avg = np.mean([float(t.amount) for t in short_transactions])
                        long_avg = features[f'avg_amount_{window_days}d']
                        
                        if long_avg > 0:
                            features[f'amount_change_ratio_{window_days}d'] = short_avg / long_avg
                        
                        features[f'frequency_change_{window_days}d'] = (
                            len(short_transactions) / short_window - 
                            len(recent_transactions) / window_days
                        )
        
        # Counterparty analysis
        features.update(self._extract_counterparty_features(historical))
        
        # Transaction type patterns
        features.update(self._extract_transaction_type_features(historical))
        
        return features
    
    def _extract_counterparty_features(self, historical: List[Transaction]) -> Dict[str, float]:
        """Extract features related to transaction counterparties"""
        
        features = {}
        
        if not historical:
            return features
        
        # Unique counterparties
        receivers = set(t.receiver_id for t in historical)
        features['unique_receivers'] = len(receivers)
        
        # Counterparty concentration
        receiver_counts = Counter(t.receiver_id for t in historical)
        if receiver_counts:
            max_count = max(receiver_counts.values())
            features['max_counterparty_ratio'] = max_count / len(historical)
            features['counterparty_entropy'] = self._calculate_entropy(
                list(receiver_counts.values())
            )
        
        # Geographic diversity
        countries = set()
        for t in historical:
            if t.destination_country:
                countries.add(t.destination_country)
            if t.origin_country:
                countries.add(t.origin_country)
        
        features['unique_countries'] = len(countries)
        
        # Cross-border ratio
        cross_border = sum(
            1 for t in historical 
            if t.origin_country != t.destination_country
        )
        features['cross_border_ratio'] = cross_border / len(historical)
        
        return features
    
    def _extract_transaction_type_features(self, historical: List[Transaction]) -> Dict[str, float]:
        """Extract features related to transaction types and patterns"""
        
        features = {}
        
        if not historical:
            return features
        
        # Asset type distribution
        asset_types = Counter(t.asset_type.value for t in historical)
        total_transactions = len(historical)
        
        for asset_type in AssetType:
            features[f'{asset_type.value}_ratio'] = (
                asset_types[asset_type.value] / total_transactions
            )
        
        # Crypto-specific features
        crypto_transactions = [t for t in historical if t.crypto_details]
        features['crypto_ratio'] = len(crypto_transactions) / total_transactions
        
        if crypto_transactions:
            # Mixer usage
            mixer_used = sum(
                1 for t in crypto_transactions 
                if t.crypto_details and t.crypto_details.mixer_used
            )
            features['mixer_usage_ratio'] = mixer_used / len(crypto_transactions)
            
            # New wallet ratio
            new_wallets = sum(
                1 for t in crypto_transactions
                if t.crypto_details and t.crypto_details.wallet_age_days < 30
            )
            features['new_wallet_ratio'] = new_wallets / len(crypto_transactions)
        
        # Document patterns
        total_documents = sum(len(t.documents) if t.documents else 0 for t in historical)
        features['avg_documents_per_transaction'] = total_documents / total_transactions
        
        # Amount patterns
        amounts = [float(t.amount) for t in historical]
        for percentile in self.config['amount_percentiles']:
            features[f'amount_p{percentile}'] = np.percentile(amounts, percentile)
        
        # Round number analysis
        round_amounts = sum(1 for amount in amounts if amount % 1000 == 0)
        features['round_amount_ratio'] = round_amounts / len(amounts)
        
        return features
    
    def _extract_network_features(
        self, 
        transaction: Transaction, 
        customer: Customer,
        network_context: Dict
    ) -> Dict[str, float]:
        """Extract network-based features"""
        
        features = {}
        
        # Basic network features
        features['sender_degree'] = network_context.get('sender_degree', 0)
        features['receiver_degree'] = network_context.get('receiver_degree', 0)
        features['sender_betweenness'] = network_context.get('sender_betweenness', 0)
        features['receiver_betweenness'] = network_context.get('receiver_betweenness', 0)
        
        # Community features
        features['same_community'] = float(
            network_context.get('sender_community') == 
            network_context.get('receiver_community', -1)
        )
        features['community_risk_score'] = network_context.get('community_risk', 0)
        
        # Path features
        features['shortest_path_length'] = network_context.get('path_length', float('inf'))
        features['is_direct_connection'] = float(features['shortest_path_length'] == 1)
        
        # Trust/reputation features
        features['sender_reputation'] = network_context.get('sender_reputation', 0.5)
        features['receiver_reputation'] = network_context.get('receiver_reputation', 0.5)
        
        return features
    
    def _extract_statistical_features(
        self, 
        transaction: Transaction, 
        historical: List[Transaction]
    ) -> Dict[str, float]:
        """Extract statistical features"""
        
        features = {}
        
        if not historical:
            return features
        
        amounts = [float(t.amount) for t in historical]
        current_amount = float(transaction.amount)
        
        # Basic statistics
        features['mean_amount'] = np.mean(amounts)
        features['std_amount'] = np.std(amounts)
        features['median_amount'] = np.median(amounts)
        features['min_amount'] = np.min(amounts)
        features['max_amount'] = np.max(amounts)
        
        # Higher order moments
        if len(amounts) > 2:
            features['skewness'] = self._calculate_skewness(amounts)
            features['kurtosis'] = self._calculate_kurtosis(amounts)
        
        # Current transaction relative to history
        if features['std_amount'] > 0:
            features['amount_zscore'] = (
                (current_amount - features['mean_amount']) / features['std_amount']
            )
        
        features['amount_percentile_rank'] = (
            sum(1 for a in amounts if a <= current_amount) / len(amounts)
        )
        
        # Outlier detection
        q1 = np.percentile(amounts, 25)
        q3 = np.percentile(amounts, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        features['is_outlier'] = float(
            current_amount < lower_bound or current_amount > upper_bound
        )
        
        # Distribution features
        features['coefficient_of_variation'] = (
            features['std_amount'] / features['mean_amount'] 
            if features['mean_amount'] > 0 else 0
        )
        
        return features
    
    def _extract_embedding_features(
        self, 
        transaction: Transaction, 
        customer: Customer,
        historical: List[Transaction]
    ) -> Dict[str, List[float]]:
        """Extract embedding-based features"""
        
        features = {}
        
        # Sequence embeddings (simplified - would use real embeddings in production)
        sequence_features = []
        
        # Recent transaction sequence
        recent_amounts = [float(t.amount) for t in historical[-10:]]
        if len(recent_amounts) < 10:
            recent_amounts.extend([0] * (10 - len(recent_amounts)))
        
        # Normalize amounts
        if recent_amounts:
            max_amount = max(recent_amounts) if max(recent_amounts) > 0 else 1
            sequence_features.extend([a / max_amount for a in recent_amounts])
        
        # Time-based embeddings
        if historical:
            time_intervals = []
            for i in range(1, min(len(historical), 6)):
                interval = (
                    historical[i].timestamp - historical[i-1].timestamp
                ).total_seconds() / 3600  # hours
                time_intervals.append(interval)
            
            # Normalize intervals
            if time_intervals:
                max_interval = max(time_intervals) if max(time_intervals) > 0 else 1
                sequence_features.extend([t / max_interval for t in time_intervals])
        
        # Pad to fixed size
        target_size = self.config['embedding_dimensions']
        if len(sequence_features) < target_size:
            sequence_features.extend([0] * (target_size - len(sequence_features)))
        else:
            sequence_features = sequence_features[:target_size]
        
        features['sequence_embedding'] = sequence_features
        
        return features
    
    def _extract_metadata(
        self, 
        transaction: Transaction, 
        customer: Customer
    ) -> Dict[str, Any]:
        """Extract metadata for feature tracking"""
        
        return {
            'transaction_id': transaction.transaction_id,
            'customer_id': customer.customer_id,
            'feature_extraction_time': datetime.utcnow().isoformat(),
            'feature_version': '1.0',
            'extraction_config': self.config
        }
    
    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate entropy of a distribution"""
        if not values:
            return 0.0
        
        # Convert to probabilities
        total = sum(values)
        if total == 0:
            return 0.0
        
        probabilities = [v / total for v in values if v > 0]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities)
        return entropy
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of a distribution"""
        if len(values) < 3:
            return 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean([((x - mean_val) / std_val) ** 3 for x in values])
        return skewness
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of a distribution"""
        if len(values) < 4:
            return 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return 0.0
        
        kurtosis = np.mean([((x - mean_val) / std_val) ** 4 for x in values]) - 3
        return kurtosis
    
    def detect_temporal_patterns(
        self, 
        transactions: List[Transaction]
    ) -> List[TemporalPattern]:
        """Detect complex temporal patterns"""
        
        patterns = []
        
        if len(transactions) < self.config['min_transactions_for_pattern']:
            return patterns
        
        # Sort by timestamp
        sorted_transactions = sorted(transactions, key=lambda x: x.timestamp)
        
        # Pattern 1: Regular intervals
        intervals = []
        for i in range(1, len(sorted_transactions)):
            interval = (
                sorted_transactions[i].timestamp - sorted_transactions[i-1].timestamp
            ).total_seconds() / 3600  # hours
            intervals.append(interval)
        
        if intervals:
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            regularity = 1 - (std_interval / avg_interval) if avg_interval > 0 else 0
            
            if regularity > 0.8:  # High regularity
                patterns.append(TemporalPattern(
                    pattern_type="regular_intervals",
                    pattern_strength=regularity,
                    time_window=f"{avg_interval:.1f} hours",
                    description=f"Regular transactions every {avg_interval:.1f} hours",
                    risk_indicator=True
                ))
        
        # Pattern 2: Burst activity
        burst_threshold = np.percentile([len(intervals)], 90) if intervals else 0
        short_intervals = [i for i in intervals if i < 1]  # < 1 hour
        
        if len(short_intervals) > burst_threshold:
            burst_strength = len(short_intervals) / len(intervals)
            patterns.append(TemporalPattern(
                pattern_type="burst_activity",
                pattern_strength=burst_strength,
                time_window="< 1 hour",
                description=f"Burst of {len(short_intervals)} transactions in short timeframe",
                risk_indicator=True
            ))
        
        # Pattern 3: Time-of-day clustering
        hours = [t.timestamp.hour for t in sorted_transactions]
        hour_counts = Counter(hours)
        
        if hour_counts:
            max_hour_count = max(hour_counts.values())
            concentration = max_hour_count / len(hours)
            
            if concentration > 0.5:  # More than 50% in same hour
                most_common_hour = hour_counts.most_common(1)[0][0]
                patterns.append(TemporalPattern(
                    pattern_type="time_clustering",
                    pattern_strength=concentration,
                    time_window=f"Hour {most_common_hour}",
                    description=f"{concentration:.1%} of transactions at hour {most_common_hour}",
                    risk_indicator=True
                ))
        
        return patterns


def create_sample_features() -> FeatureSet:
    """Create sample feature set for demonstration"""
    
    return FeatureSet(
        temporal_features={
            'hour_of_day': 14,
            'day_of_week': 1,  # Tuesday
            'is_weekend': 0.0,
            'velocity_7d': 5.2,
            'regularity_score': 0.8
        },
        behavioral_features={
            'account_age_days': 120,
            'avg_amount_30d': 15000,
            'unique_receivers': 8,
            'cross_border_ratio': 0.3
        },
        network_features={
            'sender_degree': 15,
            'community_risk_score': 0.7,
            'shortest_path_length': 2
        },
        statistical_features={
            'amount_zscore': 2.1,
            'amount_percentile_rank': 0.95,
            'is_outlier': 1.0
        },
        embedding_features={
            'sequence_embedding': [0.1, 0.8, 0.3] + [0.0] * 125  # 128-dim
        },
        metadata={
            'feature_version': '1.0',
            'extraction_time': datetime.utcnow().isoformat()
        }
    )