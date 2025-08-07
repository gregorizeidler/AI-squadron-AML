"""
Persistence Service for AML Detection System

This service handles data persistence for analysis results, customer data,
and system metrics using various storage backends.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from ..models.analysis import AMLAnalysisResult
from ..models.customer import Customer
from ..models.transaction import Transaction


class PersistenceService:
    """
    Service for persisting AML system data.
    
    Supports:
    - Analysis result storage
    - Customer data management
    - Transaction history
    - Audit trail maintenance
    - Metrics persistence
    """
    
    def __init__(self, storage_backend: str = "local"):
        """
        Initialize the persistence service.
        
        Args:
            storage_backend: Storage backend to use ('local', 'dynamodb', 's3')
        """
        self.storage_backend = storage_backend
        self.logger = self._setup_logging()
        
        # Initialize storage backend
        if storage_backend == "local":
            self.storage = LocalFileStorage()
        elif storage_backend == "dynamodb":
            self.storage = DynamoDBStorage()
        elif storage_backend == "s3":
            self.storage = S3Storage()
        else:
            raise ValueError(f"Unsupported storage backend: {storage_backend}")
        
        self.logger.info(f"Persistence service initialized with backend: {storage_backend}")
    
    async def save_analysis_result(self, result: AMLAnalysisResult) -> bool:
        """
        Save analysis result to persistent storage.
        
        Args:
            result: AML analysis result to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to dictionary for storage
            result_data = result.dict()
            result_data["saved_at"] = datetime.utcnow().isoformat()
            
            # Save to backend
            await self.storage.save_analysis_result(result.analysis_id, result_data)
            
            # Update customer analysis history
            await self._update_customer_analysis_history(result.customer_id, result.analysis_id)
            
            # Update transaction analysis mapping
            await self._update_transaction_analysis_mapping(result.transaction_id, result.analysis_id)
            
            self.logger.info(f"Analysis result saved: {result.analysis_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving analysis result: {str(e)}")
            return False
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[AMLAnalysisResult]:
        """
        Retrieve analysis result by ID.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            AML analysis result or None if not found
        """
        try:
            result_data = await self.storage.get_analysis_result(analysis_id)
            
            if result_data:
                # Convert back to AMLAnalysisResult object
                return AMLAnalysisResult.parse_obj(result_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving analysis result: {str(e)}")
            return None
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[AMLAnalysisResult]:
        """
        Synchronous version of get_analysis_result.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            AML analysis result or None if not found
        """
        import asyncio
        return asyncio.run(self.get_analysis_result(analysis_id))
    
    async def get_customer_analyses(
        self, 
        customer_id: str, 
        limit: int = 50
    ) -> List[AMLAnalysisResult]:
        """
        Get analysis results for a specific customer.
        
        Args:
            customer_id: Customer identifier
            limit: Maximum number of results to return
            
        Returns:
            List of analysis results for the customer
        """
        try:
            analysis_ids = await self.storage.get_customer_analysis_history(customer_id, limit)
            
            results = []
            for analysis_id in analysis_ids:
                result = await self.get_analysis_result(analysis_id)
                if result:
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving customer analyses: {str(e)}")
            return []
    
    async def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent analysis summaries.
        
        Args:
            limit: Maximum number of analyses to return
            
        Returns:
            List of analysis summaries
        """
        try:
            return await self.storage.get_recent_analyses(limit)
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent analyses: {str(e)}")
            return []
    
    def get_customer_risk_profile(self, customer_id: str) -> Dict[str, Any]:
        """
        Get comprehensive risk profile for a customer.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Customer risk profile
        """
        import asyncio
        return asyncio.run(self._get_customer_risk_profile_async(customer_id))
    
    async def _get_customer_risk_profile_async(self, customer_id: str) -> Dict[str, Any]:
        """Async implementation of get_customer_risk_profile"""
        try:
            # Get customer analyses
            analyses = await self.get_customer_analyses(customer_id, limit=100)
            
            if not analyses:
                return {
                    "customer_id": customer_id,
                    "total_analyses": 0,
                    "risk_summary": "No analysis history available"
                }
            
            # Calculate risk statistics
            risk_scores = [a.risk_assessment.risk_score for a in analyses]
            sar_count = len([a for a in analyses if a.sar_recommended])
            review_count = len([a for a in analyses if a.requires_human_review])
            
            # Risk level distribution
            risk_levels = {}
            for analysis in analyses:
                level = analysis.risk_assessment.risk_level.value
                risk_levels[level] = risk_levels.get(level, 0) + 1
            
            # Recent trend (last 30 days)
            recent_cutoff = datetime.utcnow() - timedelta(days=30)
            recent_analyses = [
                a for a in analyses 
                if a.processing_started and a.processing_started > recent_cutoff
            ]
            
            return {
                "customer_id": customer_id,
                "total_analyses": len(analyses),
                "sar_count": sar_count,
                "review_count": review_count,
                "risk_statistics": {
                    "average_risk_score": sum(risk_scores) / len(risk_scores),
                    "max_risk_score": max(risk_scores),
                    "min_risk_score": min(risk_scores),
                    "risk_level_distribution": risk_levels
                },
                "recent_activity": {
                    "analyses_last_30_days": len(recent_analyses),
                    "average_recent_risk": (
                        sum(a.risk_assessment.risk_score for a in recent_analyses) / len(recent_analyses)
                        if recent_analyses else 0
                    )
                },
                "last_analysis": analyses[0].processing_started.isoformat() if analyses else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting customer risk profile: {str(e)}")
            return {
                "customer_id": customer_id,
                "error": str(e)
            }
    
    async def save_customer_data(self, customer: Customer) -> bool:
        """Save customer data to persistent storage"""
        try:
            customer_data = customer.dict()
            customer_data["saved_at"] = datetime.utcnow().isoformat()
            
            await self.storage.save_customer_data(customer.customer_id, customer_data)
            
            self.logger.info(f"Customer data saved: {customer.customer_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving customer data: {str(e)}")
            return False
    
    async def save_transaction_data(self, transaction: Transaction) -> bool:
        """Save transaction data to persistent storage"""
        try:
            transaction_data = transaction.dict()
            transaction_data["saved_at"] = datetime.utcnow().isoformat()
            
            await self.storage.save_transaction_data(transaction.transaction_id, transaction_data)
            
            self.logger.info(f"Transaction data saved: {transaction.transaction_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving transaction data: {str(e)}")
            return False
    
    async def _update_customer_analysis_history(self, customer_id: str, analysis_id: str):
        """Update customer's analysis history"""
        await self.storage.add_customer_analysis(customer_id, analysis_id)
    
    async def _update_transaction_analysis_mapping(self, transaction_id: str, analysis_id: str):
        """Update transaction to analysis mapping"""
        await self.storage.add_transaction_analysis(transaction_id, analysis_id)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup persistence service logging"""
        logger = logging.getLogger("aml_system.persistence_service")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger


class StorageBackend:
    """Base class for storage backends"""
    
    async def save_analysis_result(self, analysis_id: str, data: Dict[str, Any]):
        raise NotImplementedError
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError
    
    async def get_recent_analyses(self, limit: int) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    async def save_customer_data(self, customer_id: str, data: Dict[str, Any]):
        raise NotImplementedError
    
    async def save_transaction_data(self, transaction_id: str, data: Dict[str, Any]):
        raise NotImplementedError
    
    async def get_customer_analysis_history(self, customer_id: str, limit: int) -> List[str]:
        raise NotImplementedError
    
    async def add_customer_analysis(self, customer_id: str, analysis_id: str):
        raise NotImplementedError
    
    async def add_transaction_analysis(self, transaction_id: str, analysis_id: str):
        raise NotImplementedError


class LocalFileStorage(StorageBackend):
    """Local file system storage backend"""
    
    def __init__(self):
        self.base_path = Path("data/storage")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "analyses").mkdir(exist_ok=True)
        (self.base_path / "customers").mkdir(exist_ok=True)
        (self.base_path / "transactions").mkdir(exist_ok=True)
        (self.base_path / "mappings").mkdir(exist_ok=True)
    
    async def save_analysis_result(self, analysis_id: str, data: Dict[str, Any]):
        file_path = self.base_path / "analyses" / f"{analysis_id}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        file_path = self.base_path / "analyses" / f"{analysis_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    async def get_recent_analyses(self, limit: int) -> List[Dict[str, Any]]:
        analyses_dir = self.base_path / "analyses"
        
        # Get all analysis files sorted by modification time
        files = sorted(
            analyses_dir.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        recent_analyses = []
        for file_path in files[:limit]:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Create summary
                summary = {
                    "analysis_id": data.get("analysis_id"),
                    "transaction_id": data.get("transaction_id"),
                    "customer_id": data.get("customer_id"),
                    "risk_score": data.get("risk_assessment", {}).get("risk_score", 0),
                    "final_status": data.get("final_status"),
                    "sar_recommended": data.get("sar_recommended", False),
                    "processing_started": data.get("processing_started"),
                    "saved_at": data.get("saved_at")
                }
                recent_analyses.append(summary)
                
            except Exception:
                continue  # Skip corrupted files
        
        return recent_analyses
    
    async def save_customer_data(self, customer_id: str, data: Dict[str, Any]):
        file_path = self.base_path / "customers" / f"{customer_id}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def save_transaction_data(self, transaction_id: str, data: Dict[str, Any]):
        file_path = self.base_path / "transactions" / f"{transaction_id}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def get_customer_analysis_history(self, customer_id: str, limit: int) -> List[str]:
        mapping_file = self.base_path / "mappings" / f"customer_{customer_id}_analyses.json"
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                analyses = json.load(f)
                return analyses[-limit:]  # Return most recent
        
        return []
    
    async def add_customer_analysis(self, customer_id: str, analysis_id: str):
        mapping_file = self.base_path / "mappings" / f"customer_{customer_id}_analyses.json"
        
        analyses = []
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                analyses = json.load(f)
        
        analyses.append(analysis_id)
        
        with open(mapping_file, 'w') as f:
            json.dump(analyses, f)
    
    async def add_transaction_analysis(self, transaction_id: str, analysis_id: str):
        mapping_file = self.base_path / "mappings" / f"transaction_{transaction_id}_analysis.json"
        
        with open(mapping_file, 'w') as f:
            json.dump({"transaction_id": transaction_id, "analysis_id": analysis_id}, f)


class DynamoDBStorage(StorageBackend):
    """AWS DynamoDB storage backend"""
    
    def __init__(self):
        # Mock implementation - would integrate with boto3
        pass
    
    async def save_analysis_result(self, analysis_id: str, data: Dict[str, Any]):
        print(f"DynamoDB: Would save analysis {analysis_id}")
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        print(f"DynamoDB: Would retrieve analysis {analysis_id}")
        return None


class S3Storage(StorageBackend):
    """AWS S3 storage backend"""
    
    def __init__(self):
        # Mock implementation - would integrate with boto3
        pass
    
    async def save_analysis_result(self, analysis_id: str, data: Dict[str, Any]):
        print(f"S3: Would save analysis {analysis_id}")
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        print(f"S3: Would retrieve analysis {analysis_id}")
        return None