#!/usr/bin/env python3
"""
AI Squadron AML Detection System - Main Entry Point

This is the main entry point for the AI Squadron AML Detection System.
It provides a command-line interface for running the system in various modes.
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.aml_system import AMLSystem
from models.transaction import Transaction, CryptoDetails, Document
from models.customer import Customer, CustomerProfile, BeneficialOwnership
from models.enums import AssetType, DocumentType
from services.monitoring_service import MonitoringService


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/aml_system.log')
        ]
    )


def create_sample_transaction() -> Transaction:
    """Create a sample high-risk transaction for demonstration"""
    crypto_details = CryptoDetails(
        wallet_address_from="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        wallet_address_to="3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",
        wallet_age_days=3,
        blockchain="Bitcoin",
        cryptocurrency="BTC",
        mixer_used=True,
        privacy_coin=False,
        cross_chain_swaps=2,
        darknet_market="AlphaMarket",
        exchange_used="unknown_exchange"
    )
    
    documents = [
        Document(
            document_id="DOC_001",
            document_type=DocumentType.INVOICE,
            document_content="Invoice #INV-2024-001: Consulting Services - $75,000 for strategic advisory services",
            uploaded_at=datetime.utcnow()
        ),
        Document(
            document_id="DOC_002", 
            document_type=DocumentType.CONTRACT,
            document_content="Service Agreement between ABC Corp and XYZ Holdings for blockchain consulting",
            uploaded_at=datetime.utcnow()
        )
    ]
    
    return Transaction(
        transaction_id="DEMO_TXN_001",
        timestamp=datetime.utcnow(),
        amount=Decimal("75000"),
        currency="USD",
        asset_type=AssetType.CRYPTO,
        sender_id="customer_demo_001",
        receiver_id="exchange_unknown_001",
        parties=["ABC Corp", "XYZ Holdings", "Crypto Exchange Ltd"],
        origin_country="US",
        destination_country="CH",
        intermediate_countries=["KY", "PA"],
        description="High-value cryptocurrency transaction through multiple jurisdictions",
        crypto_details=crypto_details,
        documents=documents,
        frequency=1
    )


def create_sample_customer() -> Customer:
    """Create a sample customer profile for demonstration"""
    beneficial_owners = [
        BeneficialOwnership(
            owner_name="John Smith Minister",
            ownership_percentage=51.0,
            relationship="CEO and Majority Owner",
            country_of_residence="US",
            is_pep=True,
            verified=False
        ),
        BeneficialOwnership(
            owner_name="Shell Company Holdings Ltd",
            ownership_percentage=25.0,
            relationship="Corporate Shareholder",
            country_of_residence="KY",
            is_pep=False,
            verified=False
        )
    ]
    
    profile = CustomerProfile(
        industry="Consulting Services",
        business_type="Limited Liability Company",
        annual_revenue=Decimal("2000000"),
        number_of_employees=15,
        is_cash_intensive=False,
        high_risk_industry=False,
        kyc_completed=True,
        kyc_completion_date=datetime.utcnow(),
        edd_required=True,
        edd_triggers=["PEP_DETECTED", "HIGH_RISK_JURISDICTION"]
    )
    
    return Customer(
        customer_id="customer_demo_001",
        name="ABC Consulting Corp",
        customer_type="Entity",
        account_number="ACC_001234567",
        account_age_days=45,
        account_opening_date=datetime.utcnow(),
        risk_level="MEDIUM",
        risk_score=45,
        beneficial_owners=beneficial_owners,
        profile=profile,
        total_transactions=12,
        total_transaction_volume=Decimal("500000")
    )


def run_demo():
    """Run a comprehensive demonstration of the AML system"""
    print("🛡️  Starting AI Squadron AML Detection System Demo")
    print("=" * 60)
    
    # Initialize the system
    aml_system = AMLSystem()
    aml_system.start()
    
    try:
        # Create sample data
        transaction = create_sample_transaction()
        customer = create_sample_customer()
        
        print(f"📊 Analyzing Transaction: {transaction.transaction_id}")
        print(f"💰 Amount: {transaction.amount} {transaction.currency}")
        print(f"🏦 Customer: {customer.name}")
        print(f"🌍 Route: {transaction.origin_country} → {' → '.join(transaction.intermediate_countries)} → {transaction.destination_country}")
        print(f"₿  Asset Type: {transaction.asset_type.value}")
        print("-" * 60)
        
        # Run analysis
        result = aml_system.analyze_transaction(transaction, customer)
        
        # Display results
        print("🔍 ANALYSIS RESULTS")
        print("-" * 60)
        print(f"📊 Risk Score: {result.risk_assessment.risk_score}/100")
        print(f"⚠️  Risk Level: {result.risk_assessment.risk_level.value}")
        print(f"🛑 SAR Recommended: {'✅ YES' if result.sar_recommended else '❌ NO'}")
        print(f"👤 Human Review Required: {'✅ YES' if result.requires_human_review else '❌ NO'}")
        print(f"✅ Transaction Approved: {'✅ YES' if result.transaction_approved else '❌ NO'}")
        
        if result.case_id:
            print(f"📋 Case ID: {result.case_id}")
        
        print(f"🔄 Decision Path: {' → '.join(result.decision_path)}")
        
        # Display alerts
        if result.alerts:
            print("\n🚨 ALERTS GENERATED")
            print("-" * 60)
            for alert in result.alerts:
                print(f"• {alert.alert_type.value}: {alert.description}")
                print(f"  Severity: {alert.severity.value} | Confidence: {alert.confidence_score:.1%}")
        
        # Display risk factors
        if result.risk_assessment.risk_factors:
            print("\n⚠️  RISK FACTORS IDENTIFIED")
            print("-" * 60)
            for factor in result.risk_assessment.risk_factors[:5]:  # Show top 5
                print(f"• {factor.factor_type}: {factor.factor_value} (Weight: {factor.weight})")
        
        # Display LLM analysis highlights
        if result.llm_analyses:
            print("\n🧠 AI ANALYSIS HIGHLIGHTS")
            print("-" * 60)
            for analysis in result.llm_analyses[:2]:  # Show first 2
                print(f"• {analysis.analysis_type.value}:")
                # Show first few lines of analysis
                lines = analysis.analysis_text.split('\n')[:3]
                for line in lines:
                    if line.strip():
                        print(f"  {line.strip()}")
                print()
        
        # System metrics
        metrics = aml_system.get_system_status()
        print("\n📈 SYSTEM METRICS")
        print("-" * 60)
        print(f"• Processed Transactions: {metrics['processed_transactions']}")
        print(f"• System Uptime: {metrics['uptime_seconds']:.1f} seconds")
        print(f"• Workflow Version: {metrics['workflow_info']['workflow_version']}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        logging.exception("Demo failed")
    
    finally:
        aml_system.stop()


def run_batch_demo():
    """Run a batch processing demonstration"""
    print("🔄 Starting Batch Processing Demo")
    print("=" * 60)
    
    aml_system = AMLSystem()
    aml_system.start()
    
    try:
        # Create multiple sample transactions
        transactions_and_customers = []
        
        for i in range(3):
            transaction = create_sample_transaction()
            transaction.transaction_id = f"BATCH_TXN_{i+1:03d}"
            transaction.amount = Decimal(str(25000 + (i * 15000)))
            
            customer = create_sample_customer()
            customer.customer_id = f"customer_batch_{i+1:03d}"
            customer.name = f"Demo Company {i+1}"
            
            transactions_and_customers.append((transaction, customer))
        
        # Process batch
        print(f"Processing {len(transactions_and_customers)} transactions...")
        
        async def run_batch():
            results = await aml_system.analyze_batch_async(transactions_and_customers)
            return results
        
        results = asyncio.run(run_batch())
        
        # Display batch results
        print("\n📊 BATCH RESULTS")
        print("-" * 60)
        
        total_high_risk = 0
        total_sar = 0
        total_review = 0
        
        for i, result in enumerate(results):
            print(f"Transaction {i+1}: Risk Score {result.risk_assessment.risk_score}/100")
            if result.risk_assessment.risk_score >= 75:
                total_high_risk += 1
            if result.sar_recommended:
                total_sar += 1
            if result.requires_human_review:
                total_review += 1
        
        print(f"\nSummary:")
        print(f"• High Risk Transactions: {total_high_risk}")
        print(f"• SARs Recommended: {total_sar}")
        print(f"• Requiring Review: {total_review}")
        
    except Exception as e:
        print(f"❌ Error during batch demo: {str(e)}")
        logging.exception("Batch demo failed")
    
    finally:
        aml_system.stop()


def run_monitoring_demo():
    """Run a monitoring system demonstration"""
    print("📊 Starting Monitoring System Demo")
    print("=" * 60)
    
    monitoring = MonitoringService()
    monitoring.start()
    
    try:
        # Simulate some metrics
        for i in range(5):
            # Create dummy analysis result
            result = create_sample_transaction()
            # monitoring.record_analysis_completed(result)  # Would need proper result object
        
        # Display metrics
        metrics = monitoring.get_metrics()
        health = monitoring.get_system_health()
        
        print("📈 SYSTEM METRICS")
        print("-" * 60)
        print(f"• System Status: {health['status']}")
        print(f"• Total Analyses: {metrics['performance']['total_analyses']}")
        print(f"• Average Processing Time: {metrics['performance']['average_processing_time']:.2f}s")
        print(f"• Throughput: {metrics['performance']['throughput_per_hour']:.1f} per hour")
        print(f"• Error Rate: {health['error_rate']:.1%}")
        
        # Generate summary report
        report = monitoring.generate_summary_report()
        print("\n📋 MONITORING REPORT")
        print("-" * 60)
        print(report)
        
    except Exception as e:
        print(f"❌ Error during monitoring demo: {str(e)}")
        logging.exception("Monitoring demo failed")
    
    finally:
        monitoring.stop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI Squadron AML Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --demo              # Run interactive demo
  python -m src.main --batch             # Run batch processing demo  
  python -m src.main --monitoring        # Run monitoring demo
  python -m src.main --log-level DEBUG   # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run interactive demonstration'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true', 
        help='Run batch processing demonstration'
    )
    
    parser.add_argument(
        '--monitoring',
        action='store_true',
        help='Run monitoring system demonstration'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    if args.demo:
        run_demo()
    elif args.batch:
        run_batch_demo()
    elif args.monitoring:
        run_monitoring_demo()
    else:
        print("🛡️  AI Squadron AML Detection System")
        print("Please specify a mode: --demo, --batch, or --monitoring")
        print("Use --help for more information")


if __name__ == "__main__":
    main()