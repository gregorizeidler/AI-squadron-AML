"""
Customer-related data models for the AML detection system
"""
from datetime import datetime, date
from decimal import Decimal
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from .enums import RiskLevel, CountryRiskLevel


class CustomerAddress(BaseModel):
    """Customer address information"""
    address_line_1: str = Field(..., description="Primary address line")
    address_line_2: Optional[str] = Field(None, description="Secondary address line")
    city: str = Field(..., description="City")
    state_province: Optional[str] = Field(None, description="State or province")
    postal_code: Optional[str] = Field(None, description="Postal/ZIP code")
    country: str = Field(..., description="Country (ISO 2-letter code)")
    is_primary: bool = Field(True, description="Whether this is the primary address")
    
    @validator('country')
    def validate_country_code(cls, v):
        """Validate country code"""
        if len(v) != 2:
            raise ValueError("Country code must be 2-letter ISO code")
        return v.upper()


class BeneficialOwnership(BaseModel):
    """Beneficial ownership information"""
    owner_name: str = Field(..., description="Name of beneficial owner")
    ownership_percentage: float = Field(..., description="Percentage of ownership", ge=0, le=100)
    relationship: str = Field(..., description="Relationship to customer")
    id_number: Optional[str] = Field(None, description="Identification number")
    country_of_residence: str = Field(..., description="Country of residence")
    is_pep: bool = Field(False, description="Whether owner is a PEP")
    is_sanctioned: bool = Field(False, description="Whether owner is sanctioned")
    verified: bool = Field(False, description="Whether ownership is verified")
    verification_date: Optional[datetime] = Field(None, description="Date of verification")
    
    @validator('country_of_residence')
    def validate_country(cls, v):
        """Validate country code"""
        if len(v) != 2:
            raise ValueError("Country code must be 2-letter ISO code")
        return v.upper()


class CustomerProfile(BaseModel):
    """Extended customer profile information"""
    industry: Optional[str] = Field(None, description="Industry sector")
    business_type: Optional[str] = Field(None, description="Type of business")
    annual_revenue: Optional[Decimal] = Field(None, description="Annual revenue")
    number_of_employees: Optional[int] = Field(None, description="Number of employees")
    incorporation_date: Optional[date] = Field(None, description="Date of incorporation")
    incorporation_country: Optional[str] = Field(None, description="Country of incorporation")
    
    # Risk factors
    is_cash_intensive: bool = Field(False, description="Whether business is cash-intensive")
    high_risk_industry: bool = Field(False, description="Whether in high-risk industry")
    shell_company_indicators: List[str] = Field(default_factory=list, description="Shell company risk indicators")
    
    # Compliance information
    kyc_completed: bool = Field(False, description="Whether KYC is completed")
    kyc_completion_date: Optional[datetime] = Field(None, description="KYC completion date")
    kyc_next_review: Optional[datetime] = Field(None, description="Next KYC review date")
    
    # Enhanced due diligence
    edd_required: bool = Field(False, description="Whether EDD is required")
    edd_completion_date: Optional[datetime] = Field(None, description="EDD completion date")
    edd_triggers: List[str] = Field(default_factory=list, description="EDD trigger reasons")


class Customer(BaseModel):
    """Core customer model"""
    customer_id: str = Field(..., description="Unique customer identifier")
    name: str = Field(..., description="Customer name or business name")
    customer_type: str = Field(..., description="Individual or Entity")
    
    # Basic information
    date_of_birth: Optional[date] = Field(None, description="Date of birth (individuals)")
    nationality: Optional[str] = Field(None, description="Nationality")
    identification_number: Optional[str] = Field(None, description="ID/Tax number")
    identification_type: Optional[str] = Field(None, description="Type of identification")
    
    # Contact information
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    addresses: List[CustomerAddress] = Field(default_factory=list, description="Customer addresses")
    
    # Account information
    account_number: str = Field(..., description="Account number")
    account_age_days: int = Field(..., description="Age of account in days")
    account_opening_date: datetime = Field(..., description="Account opening date")
    account_status: str = Field(default="ACTIVE", description="Account status")
    
    # Risk assessment
    risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="Customer risk level")
    risk_score: int = Field(default=0, description="Numerical risk score", ge=0, le=100)
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    
    # PEP and sanctions status
    is_pep: bool = Field(False, description="Whether customer is a PEP")
    pep_category: Optional[str] = Field(None, description="PEP category if applicable")
    is_sanctioned: bool = Field(False, description="Whether customer is sanctioned")
    sanctions_lists: List[str] = Field(default_factory=list, description="Applicable sanctions lists")
    
    # Beneficial ownership
    beneficial_owners: List[BeneficialOwnership] = Field(
        default_factory=list, 
        description="Beneficial ownership information"
    )
    
    # Customer profile
    profile: Optional[CustomerProfile] = Field(None, description="Extended customer profile")
    
    # Transaction history summary
    total_transactions: int = Field(default=0, description="Total number of transactions")
    total_transaction_volume: Decimal = Field(default=Decimal("0"), description="Total transaction volume")
    average_transaction_amount: Decimal = Field(default=Decimal("0"), description="Average transaction amount")
    last_transaction_date: Optional[datetime] = Field(None, description="Date of last transaction")
    
    # Monitoring and alerts
    monitoring_level: str = Field(default="STANDARD", description="Monitoring level")
    alert_count: int = Field(default=0, description="Number of alerts generated")
    last_alert_date: Optional[datetime] = Field(None, description="Date of last alert")
    
    # Compliance dates
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation date")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update date")
    last_reviewed: Optional[datetime] = Field(None, description="Last review date")
    next_review_due: Optional[datetime] = Field(None, description="Next review due date")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional customer metadata")
    
    @validator('nationality')
    def validate_nationality(cls, v):
        """Validate nationality code"""
        if v and len(v) != 2:
            raise ValueError("Nationality must be 2-letter ISO code")
        return v.upper() if v else v
    
    @validator('risk_score')
    def validate_risk_score(cls, v):
        """Validate risk score range"""
        if not 0 <= v <= 100:
            raise ValueError("Risk score must be between 0 and 100")
        return v
    
    @validator('account_age_days')
    def validate_account_age(cls, v):
        """Validate account age"""
        if v < 0:
            raise ValueError("Account age cannot be negative")
        return v
    
    def get_primary_address(self) -> Optional[CustomerAddress]:
        """Get the primary address"""
        for address in self.addresses:
            if address.is_primary:
                return address
        return self.addresses[0] if self.addresses else None
    
    def calculate_risk_level(self) -> RiskLevel:
        """Calculate risk level based on risk score"""
        if self.risk_score >= 75:
            return RiskLevel.CRITICAL
        elif self.risk_score >= 50:
            return RiskLevel.HIGH
        elif self.risk_score >= 25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def is_high_risk(self) -> bool:
        """Check if customer is high risk"""
        return self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    def requires_edd(self) -> bool:
        """Check if customer requires enhanced due diligence"""
        return (
            self.is_pep or 
            self.is_sanctioned or 
            self.is_high_risk() or
            (self.profile and self.profile.edd_required)
        )
    
    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }