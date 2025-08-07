"""
Graph Neural Network for AML Detection

Advanced graph-based analysis using neural networks to detect
money laundering patterns in transaction networks.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from decimal import Decimal
import networkx as nx
from collections import defaultdict

# Would use these in production:
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, SAGEConv, GATConv
# from torch_geometric.data import Data, DataLoader
# import community as community_louvain  # python-louvain
# from sklearn.cluster import DBSCAN, SpectralClustering

from ..models.transaction import Transaction
from ..models.customer import Customer
from ..models.enums import RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the transaction graph"""
    node_id: str
    node_type: str  # 'customer', 'account', 'wallet', 'entity'
    risk_score: float
    features: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass 
class GraphEdge:
    """Represents an edge (transaction) in the graph"""
    source: str
    target: str
    weight: float  # transaction amount or frequency
    edge_type: str  # 'transfer', 'deposit', 'withdrawal'
    timestamp: datetime
    features: Dict[str, Any]


@dataclass
class Community:
    """Represents a detected community in the graph"""
    community_id: str
    nodes: Set[str]
    risk_level: RiskLevel
    suspicion_score: float
    community_type: str  # 'structuring', 'layering', 'integration'
    characteristics: Dict[str, Any]


@dataclass
class GraphAnalysisResult:
    """Results from graph neural network analysis"""
    node_risk_scores: Dict[str, float]
    edge_risk_scores: Dict[Tuple[str, str], float]
    communities: List[Community]
    suspicious_patterns: List[Dict[str, Any]]
    network_metrics: Dict[str, float]
    recommendations: List[str]


class CommunityDetector:
    """Advanced community detection for money laundering networks"""
    
    def __init__(self):
        self.algorithms = {
            'louvain': self._louvain_detection,
            'leiden': self._leiden_detection,
            'spectral': self._spectral_clustering,
            'dbscan': self._dbscan_clustering
        }
        
    def detect_communities(
        self, 
        graph: nx.Graph, 
        algorithm: str = 'louvain'
    ) -> List[Community]:
        """Detect communities using specified algorithm"""
        try:
            if algorithm not in self.algorithms:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            communities = self.algorithms[algorithm](graph)
            analyzed_communities = []
            
            for i, community_nodes in enumerate(communities):
                community = self._analyze_community(
                    graph, community_nodes, f"COMM_{i:03d}"
                )
                analyzed_communities.append(community)
            
            return analyzed_communities
            
        except Exception as e:
            logger.error(f"Community detection failed: {str(e)}")
            return []
    
    def _louvain_detection(self, graph: nx.Graph) -> List[Set[str]]:
        """Louvain algorithm for community detection"""
        # In production, would use: community_louvain.best_partition(graph)
        # For demo, using simple connected components
        communities = []
        for component in nx.connected_components(graph):
            if len(component) >= 3:  # Minimum community size
                communities.append(component)
        return communities
    
    def _leiden_detection(self, graph: nx.Graph) -> List[Set[str]]:
        """Leiden algorithm (improved Louvain)"""
        # Would use leidenalg library in production
        return self._louvain_detection(graph)  # Fallback for demo
    
    def _spectral_clustering(self, graph: nx.Graph) -> List[Set[str]]:
        """Spectral clustering for community detection"""
        # Would use sklearn.cluster.SpectralClustering
        return self._louvain_detection(graph)  # Fallback for demo
    
    def _dbscan_clustering(self, graph: nx.Graph) -> List[Set[str]]:
        """DBSCAN clustering based on graph features"""
        # Would use sklearn.cluster.DBSCAN with graph features
        return self._louvain_detection(graph)  # Fallback for demo
    
    def _analyze_community(
        self, 
        graph: nx.Graph, 
        nodes: Set[str], 
        community_id: str
    ) -> Community:
        """Analyze community characteristics and risk"""
        
        # Calculate community metrics
        subgraph = graph.subgraph(nodes)
        density = nx.density(subgraph)
        clustering = nx.average_clustering(subgraph) if len(nodes) > 2 else 0
        
        # Analyze transaction patterns
        total_value = 0
        transaction_count = 0
        unique_days = set()
        
        for edge in subgraph.edges(data=True):
            if 'weight' in edge[2]:
                total_value += edge[2]['weight']
                transaction_count += 1
            if 'timestamp' in edge[2]:
                unique_days.add(edge[2]['timestamp'].date())
        
        # Calculate suspicion score
        suspicion_score = self._calculate_community_suspicion(
            nodes, density, clustering, total_value, transaction_count, len(unique_days)
        )
        
        # Determine community type
        community_type = self._classify_community_type(
            density, clustering, total_value, transaction_count
        )
        
        # Determine risk level
        if suspicion_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif suspicion_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif suspicion_score >= 0.4:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        characteristics = {
            'size': len(nodes),
            'density': density,
            'clustering_coefficient': clustering,
            'total_transaction_value': total_value,
            'transaction_count': transaction_count,
            'active_days': len(unique_days),
            'average_transaction_value': total_value / max(transaction_count, 1)
        }
        
        return Community(
            community_id=community_id,
            nodes=nodes,
            risk_level=risk_level,
            suspicion_score=suspicion_score,
            community_type=community_type,
            characteristics=characteristics
        )
    
    def _calculate_community_suspicion(
        self, 
        nodes: Set[str], 
        density: float, 
        clustering: float,
        total_value: float, 
        transaction_count: int, 
        active_days: int
    ) -> float:
        """Calculate suspicion score for a community"""
        
        score = 0.0
        
        # High density indicates tight connections
        if density > 0.7:
            score += 0.3
        elif density > 0.5:
            score += 0.2
        
        # High clustering suggests coordinated activity
        if clustering > 0.8:
            score += 0.25
        elif clustering > 0.6:
            score += 0.15
        
        # Large communities are more suspicious
        if len(nodes) > 10:
            score += 0.2
        elif len(nodes) > 5:
            score += 0.1
        
        # High transaction volume
        if total_value > 1000000:
            score += 0.15
        elif total_value > 100000:
            score += 0.1
        
        # High frequency (many transactions in few days)
        if active_days > 0:
            frequency = transaction_count / active_days
            if frequency > 10:
                score += 0.1
        
        return min(score, 1.0)
    
    def _classify_community_type(
        self, 
        density: float, 
        clustering: float, 
        total_value: float, 
        transaction_count: int
    ) -> str:
        """Classify the type of money laundering pattern"""
        
        if density > 0.7 and clustering > 0.8:
            return "structuring"  # Tight, coordinated network
        elif total_value > 500000 and transaction_count < 10:
            return "layering"     # High-value, low-frequency
        elif transaction_count > 50:
            return "integration"  # High-frequency activity
        else:
            return "unknown"


class GraphNeuralNetwork:
    """Graph Neural Network for AML detection"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.community_detector = CommunityDetector()
        self.node_embeddings = {}
        self.is_trained = False
        
    def _default_config(self) -> Dict:
        """Default configuration for GNN"""
        return {
            'embedding_dim': 128,
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.2,
            'learning_rate': 0.01,
            'batch_size': 64,
            'epochs': 100,
            'node_features': ['degree', 'betweenness', 'closeness', 'pagerank'],
            'edge_features': ['weight', 'frequency', 'temporal_distance']
        }
    
    def build_transaction_graph(
        self, 
        transactions: List[Transaction], 
        customers: List[Customer]
    ) -> nx.Graph:
        """Build transaction graph from data"""
        
        graph = nx.Graph()
        
        # Add customer nodes
        for customer in customers:
            features = self._extract_customer_features(customer)
            graph.add_node(
                customer.customer_id,
                node_type='customer',
                features=features,
                risk_score=customer.risk_score or 0
            )
        
        # Add transaction edges
        for transaction in transactions:
            edge_features = self._extract_transaction_features(transaction)
            
            graph.add_edge(
                transaction.sender_id,
                transaction.receiver_id,
                weight=float(transaction.amount),
                edge_type='transfer',
                timestamp=transaction.timestamp,
                features=edge_features,
                transaction_id=transaction.transaction_id
            )
        
        # Calculate network features
        self._calculate_network_features(graph)
        
        return graph
    
    def _extract_customer_features(self, customer: Customer) -> Dict[str, Any]:
        """Extract features from customer data"""
        return {
            'account_age_days': customer.account_age_days,
            'total_transactions': customer.total_transactions,
            'total_volume': float(customer.total_transaction_volume or 0),
            'risk_score': customer.risk_score or 0,
            'is_pep': getattr(customer, 'is_pep', False),
            'has_beneficial_owners': len(customer.beneficial_owners) > 0 if hasattr(customer, 'beneficial_owners') else False
        }
    
    def _extract_transaction_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Extract features from transaction data"""
        return {
            'amount': float(transaction.amount),
            'currency': transaction.currency,
            'cross_border': transaction.origin_country != transaction.destination_country,
            'intermediate_countries': len(transaction.intermediate_countries),
            'crypto_transaction': transaction.crypto_details is not None,
            'document_count': len(transaction.documents) if transaction.documents else 0,
            'hour_of_day': transaction.timestamp.hour,
            'day_of_week': transaction.timestamp.weekday()
        }
    
    def _calculate_network_features(self, graph: nx.Graph):
        """Calculate advanced network features for nodes"""
        
        # Centrality measures
        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        pagerank = nx.pagerank(graph)
        
        # Update node features
        for node in graph.nodes():
            graph.nodes[node]['degree_centrality'] = degree_centrality[node]
            graph.nodes[node]['betweenness_centrality'] = betweenness_centrality[node]
            graph.nodes[node]['closeness_centrality'] = closeness_centrality[node]
            graph.nodes[node]['pagerank'] = pagerank[node]
    
    def analyze_graph(
        self, 
        transactions: List[Transaction], 
        customers: List[Customer]
    ) -> GraphAnalysisResult:
        """Comprehensive graph analysis"""
        
        # Build transaction graph
        graph = self.build_transaction_graph(transactions, customers)
        
        # Detect communities
        communities = self.community_detector.detect_communities(graph)
        
        # Calculate node risk scores
        node_risk_scores = self._calculate_node_risks(graph)
        
        # Calculate edge risk scores
        edge_risk_scores = self._calculate_edge_risks(graph)
        
        # Detect suspicious patterns
        suspicious_patterns = self._detect_suspicious_patterns(graph, communities)
        
        # Calculate network metrics
        network_metrics = self._calculate_network_metrics(graph)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            communities, suspicious_patterns, network_metrics
        )
        
        return GraphAnalysisResult(
            node_risk_scores=node_risk_scores,
            edge_risk_scores=edge_risk_scores,
            communities=communities,
            suspicious_patterns=suspicious_patterns,
            network_metrics=network_metrics,
            recommendations=recommendations
        )
    
    def _calculate_node_risks(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate risk scores for each node"""
        node_risks = {}
        
        for node in graph.nodes(data=True):
            node_id, features = node
            
            risk_score = 0.0
            
            # Base risk from customer data
            if 'risk_score' in features:
                risk_score += features['risk_score'] * 0.3
            
            # Centrality-based risk
            if 'betweenness_centrality' in features:
                risk_score += features['betweenness_centrality'] * 0.2
            
            if 'pagerank' in features:
                risk_score += features['pagerank'] * 0.2
            
            # Network position risk
            degree = graph.degree(node_id)
            if degree > 10:  # Hub nodes are more suspicious
                risk_score += 0.2
            
            # PEP status
            if features.get('is_pep', False):
                risk_score += 0.1
            
            node_risks[node_id] = min(risk_score, 1.0)
        
        return node_risks
    
    def _calculate_edge_risks(self, graph: nx.Graph) -> Dict[Tuple[str, str], float]:
        """Calculate risk scores for each edge"""
        edge_risks = {}
        
        for edge in graph.edges(data=True):
            source, target, features = edge
            
            risk_score = 0.0
            
            # High-value transactions
            weight = features.get('weight', 0)
            if weight > 100000:
                risk_score += 0.3
            elif weight > 50000:
                risk_score += 0.2
            
            # Cross-border risk
            if features.get('cross_border', False):
                risk_score += 0.1
            
            # Crypto transactions
            if features.get('crypto_transaction', False):
                risk_score += 0.2
            
            # Multiple intermediate countries
            intermediate = features.get('intermediate_countries', 0)
            if intermediate > 2:
                risk_score += 0.1
            
            edge_risks[(source, target)] = min(risk_score, 1.0)
        
        return edge_risks
    
    def _detect_suspicious_patterns(
        self, 
        graph: nx.Graph, 
        communities: List[Community]
    ) -> List[Dict[str, Any]]:
        """Detect suspicious patterns in the graph"""
        
        patterns = []
        
        # Pattern 1: Star-like structures (potential structuring)
        for node in graph.nodes():
            degree = graph.degree(node)
            if degree > 8:  # High-degree nodes
                neighbors = list(graph.neighbors(node))
                # Check if neighbors are mostly disconnected
                neighbor_connections = 0
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        if graph.has_edge(n1, n2):
                            neighbor_connections += 1
                
                # If few connections between neighbors, it's star-like
                max_connections = len(neighbors) * (len(neighbors) - 1) / 2
                if neighbor_connections / max_connections < 0.2:
                    patterns.append({
                        'type': 'star_structure',
                        'center_node': node,
                        'degree': degree,
                        'suspicion_score': 0.8,
                        'description': f'Star-like structure with {degree} connections'
                    })
        
        # Pattern 2: Circular patterns (potential layering)
        cycles = []
        for node in graph.nodes():
            try:
                # Find cycles of length 3-6
                for cycle_length in range(3, 7):
                    simple_cycles = nx.simple_cycles(
                        graph.to_directed(), length_bound=cycle_length
                    )
                    for cycle in simple_cycles:
                        if len(cycle) >= 3 and node in cycle:
                            cycles.append(cycle)
                            break
            except:
                continue
        
        for cycle in cycles[:5]:  # Limit to top 5 cycles
            patterns.append({
                'type': 'circular_pattern',
                'nodes': cycle,
                'length': len(cycle),
                'suspicion_score': 0.7,
                'description': f'Circular transaction pattern with {len(cycle)} nodes'
            })
        
        # Pattern 3: High-risk communities
        for community in communities:
            if community.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                patterns.append({
                    'type': 'high_risk_community',
                    'community_id': community.community_id,
                    'nodes': list(community.nodes),
                    'suspicion_score': community.suspicion_score,
                    'description': f'High-risk community: {community.community_type}'
                })
        
        return patterns
    
    def _calculate_network_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate overall network metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = graph.number_of_nodes()
        metrics['num_edges'] = graph.number_of_edges()
        metrics['density'] = nx.density(graph)
        
        # Centralization metrics
        if graph.number_of_nodes() > 1:
            metrics['average_clustering'] = nx.average_clustering(graph)
            
            # Average path length (for largest connected component)
            largest_cc = max(nx.connected_components(graph), key=len)
            if len(largest_cc) > 1:
                subgraph = graph.subgraph(largest_cc)
                metrics['average_path_length'] = nx.average_shortest_path_length(subgraph)
        
        # Risk distribution
        total_risk = sum(
            node_data.get('risk_score', 0) 
            for _, node_data in graph.nodes(data=True)
        )
        metrics['average_node_risk'] = total_risk / max(graph.number_of_nodes(), 1)
        
        # Transaction volume metrics
        total_volume = sum(
            edge_data.get('weight', 0) 
            for _, _, edge_data in graph.edges(data=True)
        )
        metrics['total_transaction_volume'] = total_volume
        metrics['average_transaction_value'] = total_volume / max(graph.number_of_edges(), 1)
        
        return metrics
    
    def _generate_recommendations(
        self, 
        communities: List[Community], 
        patterns: List[Dict[str, Any]], 
        metrics: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Community-based recommendations
        high_risk_communities = [
            c for c in communities 
            if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]
        
        if high_risk_communities:
            recommendations.append(
                f"Investigate {len(high_risk_communities)} high-risk communities immediately"
            )
        
        # Pattern-based recommendations
        star_patterns = [p for p in patterns if p['type'] == 'star_structure']
        if star_patterns:
            recommendations.append(
                f"Review {len(star_patterns)} potential structuring patterns"
            )
        
        circular_patterns = [p for p in patterns if p['type'] == 'circular_pattern']
        if circular_patterns:
            recommendations.append(
                f"Analyze {len(circular_patterns)} circular transaction patterns for layering"
            )
        
        # Network-level recommendations
        if metrics.get('density', 0) > 0.1:
            recommendations.append(
                "High network density detected - review for coordinated activity"
            )
        
        if metrics.get('average_node_risk', 0) > 0.6:
            recommendations.append(
                "Elevated average risk scores - implement enhanced monitoring"
            )
        
        if not recommendations:
            recommendations.append("No immediate suspicious patterns detected")
        
        return recommendations


def create_sample_graph_analysis() -> GraphAnalysisResult:
    """Create a sample graph analysis for demonstration"""
    
    # Sample communities
    communities = [
        Community(
            community_id="COMM_001",
            nodes={"customer_001", "customer_002", "customer_003"},
            risk_level=RiskLevel.HIGH,
            suspicion_score=0.85,
            community_type="structuring",
            characteristics={
                'size': 3,
                'density': 0.9,
                'total_transaction_value': 250000,
                'transaction_count': 15
            }
        )
    ]
    
    # Sample patterns
    patterns = [
        {
            'type': 'star_structure',
            'center_node': 'customer_001',
            'degree': 8,
            'suspicion_score': 0.8,
            'description': 'Star-like structure with 8 connections'
        }
    ]
    
    return GraphAnalysisResult(
        node_risk_scores={'customer_001': 0.9, 'customer_002': 0.7},
        edge_risk_scores={('customer_001', 'customer_002'): 0.8},
        communities=communities,
        suspicious_patterns=patterns,
        network_metrics={
            'num_nodes': 10,
            'num_edges': 15,
            'density': 0.3,
            'average_node_risk': 0.6
        },
        recommendations=[
            "Investigate 1 high-risk community immediately",
            "Review 1 potential structuring pattern"
        ]
    )