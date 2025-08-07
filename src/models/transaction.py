"""
Transaction-related data models for the AML detection system
"""
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from .enums import AssetType, TransactionStatus, DocumentType


class CryptoDetails(BaseModel):
    """Cryptocurrency-specific transaction details"""
    wallet_address_from: Optional[str] = Field(None, description="Source wallet address")
    wallet_address_to: Optional[str] = Field(None, description="Destination wallet address")
    wallet_age_days: Optional[int] = Field(None, description="Age of wallet in days")
    blockchain: Optional[str] = Field(None, description="Blockchain network")
    cryptocurrency: Optional[str] = Field(None, description="Cryptocurrency type")
    mixer_used: bool = Field(False, description="Whether a crypto mixer was used")
    privacy_coin: bool = Field(False, description="Whether it's a privacy coin")
    cross_chain_swaps: int = Field(0, description="Number of cross-chain swaps")
    darknet_market: Optional[str] = Field(None, description="Associated darknet market")
    mining_pool: Optional[str] = Field(None, description="Associated mining pool")
    exchange_used: Optional[str] = Field(None, description="Cryptocurrency exchange")
    
    class Config:
        extra = "allow"


class Document(BaseModel):
    """Document attached to transaction"""
    document_id: str = Field(..., description="Unique document identifier")
    document_type: DocumentType = Field(..., description="Type of document")
    document_content: str = Field(..., description="Document content or description")
    file_path: Optional[str] = Field(None, description="Path to document file")
    hash_value: Optional[str] = Field(None, description="Document hash for integrity")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        extra = "allow"


class Transaction(BaseModel):
    """Core transaction model"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Transaction timestamp")
    amount: Decimal = Field(..., description="Transaction amount", gt=0)
    currency: str = Field(default="USD", description="Transaction currency")
    asset_type: AssetType = Field(default=AssetType.FIAT, description="Type of asset being transacted")
    
    # Parties
    sender_id: str = Field(..., description="Sender identifier")
    receiver_id: str = Field(..., description="Receiver identifier")
    parties: List[str] = Field(default_factory=list, description="All parties involved")
    
    # Geographic information
    origin_country: str = Field(..., description="Country of origin (ISO 2-letter code)")
    destination_country: str = Field(..., description="Destination country (ISO 2-letter code)")
    intermediate_countries: List[str] = Field(default_factory=list, description="Intermediate countries")
    
    # Transaction details
    description: Optional[str] = Field(None, description="Transaction description")
    reference_number: Optional[str] = Field(None, description="External reference number")
    transaction_type: Optional[str] = Field(None, description="Type of transaction")
    
    # Status and processing
    status: TransactionStatus = Field(default=TransactionStatus.PENDING)
    processed_at: Optional[datetime] = Field(None, description="When transaction was processed")
    
    # Supporting documents
    documents: List[Document] = Field(default_factory=list, description="Supporting documents")
    
    # Cryptocurrency-specific details
    crypto_details: Optional[CryptoDetails] = Field(None, description="Crypto-specific details")
    
    # Frequency and pattern information
    frequency: int = Field(default=1, description="Transaction frequency (for patterns)")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional transaction metadata")
    
    @validator('amount')
    def validate_amount(cls, v):
        """Validate transaction amount"""
        if v <= 0:
            raise ValueError("Transaction amount must be positive")
        return v
    
    @validator('origin_country', 'destination_country')
    def validate_country_codes(cls, v):
        """Validate country codes are 2 characters"""
        if len(v) != 2:
            raise ValueError("Country codes must be 2-letter ISO codes")
        return v.upper()
    
    @validator('intermediate_countries')
    def validate_intermediate_countries(cls, v):
        """Validate intermediate country codes"""
        for country in v:
            if len(country) != 2:
                raise ValueError("All country codes must be 2-letter ISO codes")
        return [c.upper() for c in v]
    
    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }


class TransactionHistory(BaseModel):
    """Historical transaction data for a customer"""
    customer_id: str = Field(..., description="Customer identifier")
    transactions: List[Transaction] = Field(..., description="List of historical transactions")
    total_transactions: int = Field(..., description="Total number of transactions")
    total_volume: Decimal = Field(..., description="Total transaction volume")
    date_range_start: datetime = Field(..., description="Start of historical period")
    date_range_end: datetime = Field(..., description="End of historical period")
    
    # Aggregated statistics
    average_transaction_amount: Decimal = Field(..., description="Average transaction amount")
    largest_transaction: Decimal = Field(..., description="Largest transaction amount")
    countries_involved: List[str] = Field(..., description="All countries involved in transactions")
    asset_types_used: List[AssetType] = Field(..., description="Asset types used")
    
    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }


class TransactionPattern(BaseModel):
    """Detected transaction pattern"""
    pattern_id: str = Field(..., description="Unique pattern identifier")
    pattern_type: str = Field(..., description="Type of pattern detected")
    confidence_score: float = Field(..., description="Confidence in pattern detection", ge=0, le=1)
    transactions: List[str] = Field(..., description="Transaction IDs involved in pattern")
    description: str = Field(..., description="Pattern description")
    risk_indicators: List[str] = Field(..., description="Associated risk indicators")
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        extra = "allow"