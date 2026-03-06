#!/usr/bin/env python3
"""
Integration test for analysis tools.
Tests with real ClickHouse data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.uds.tools import (
    SalesTrendTool,
    InventoryAnalysisTool,
    ProductPerformanceTool,
    FinancialSummaryTool,
    ComparisonTool
)

def test_sales_trend():
    """Test sales trend analysis."""
    print("\n" + "="*80)
    print("SALES TREND ANALYSIS TEST")
    print("="*80)
    
    tool = SalesTrendTool()
    
    test_cases = [
        ("Full October", "2025-10-01", "2025-10-31"),
        ("First week", "2025-10-01", "2025-10-07"),
        ("Mid-month", "2025-10-15", "2025-10-20"),
    ]
    
    for i, (name, start, end) in enumerate(test_cases, 1):
        print(f"\n{i}. Test: {name} ({start} to {end})")
        result = tool.execute(start_date=start, end_date=end)
        
        if result.success:
            insights = result.output['insights']
            print(f"   ✅ Analysis successful")
            print(f"      Total Revenue: ${insights['total_revenue']:,.2f}")
            print(f"      Total Orders: {insights['total_orders']:,}")
            print(f"      Growth Rate: {insights.get('growth_rate_pct', 0):.2f}%")
            print(f"      Trend: {insights.get('trend', 'N/A')}")
        else:
            print(f"   ❌ Failed: {result.error}")
            return False
    
    return True


def test_inventory_analysis():
    """Test inventory analysis."""
    print("\n" + "="*80)
    print("INVENTORY ANALYSIS TEST")
    print("="*80)
    
    tool = InventoryAnalysisTool()
    
    test_cases = [
        ("Default threshold (10)", None, 10),
        ("High threshold (50)", None, 50),
        ("Specific date", "2025-10-15", 10),
    ]
    
    for i, (name, date, threshold) in enumerate(test_cases, 1):
        print(f"\n{i}. Test: {name}")
        result = tool.execute(as_of_date=date, low_stock_threshold=threshold)
        
        if result.success:
            summary = result.output['inventory_summary']
            alerts = result.output['alerts']
            print(f"   ✅ Analysis successful")
            print(f"      Total SKUs: {summary['total_skus']:,}")
            print(f"      Total Units: {summary['total_units']:,}")
            print(f"      Low Stock Items: {summary['low_stock_items']}")
            if alerts:
                print(f"      Alerts: {len(alerts)}")
                for alert in alerts[:2]:
                    print(f"        - {alert}")
        else:
            print(f"   ❌ Failed: {result.error}")
            return False
    
    return True


def test_product_performance():
    """Test product performance analysis."""
    print("\n" + "="*80)
    print("PRODUCT PERFORMANCE TEST")
    print("="*80)
    
    tool = ProductPerformanceTool()
    
    test_cases = [
        ("Top 10 by revenue", "2025-10-01", "2025-10-31", "revenue", 10),
        ("Top 5 by units", "2025-10-01", "2025-10-31", "units", 5),
        ("Weekly analysis", "2025-10-01", "2025-10-07", "revenue", 10),
    ]
    
    for i, (name, start, end, metric, limit) in enumerate(test_cases, 1):
        print(f"\n{i}. Test: {name}")
        result = tool.execute(
            start_date=start,
            end_date=end,
            metric=metric,
            limit=limit
        )
        
        if result.success:
            insights = result.output['insights']
            top_products = result.output['top_products']
            print(f"   ✅ Analysis successful")
            print(f"      Products Analyzed: {insights['total_products']}")
            print(f"      Total Revenue: ${insights['total_revenue']:,.2f}")
            print(f"      Total Units: {insights['total_units']:,}")
            print(f"      Top Product: {insights['top_product']['name']}")
            print(f"      Revenue Concentration: {insights['revenue_concentration']:.1f}%")
        else:
            print(f"   ❌ Failed: {result.error}")
            return False
    
    return True


def test_financial_summary():
    """Test financial summary."""
    print("\n" + "="*80)
    print("FINANCIAL SUMMARY TEST")
    print("="*80)
    
    tool = FinancialSummaryTool()
    
    test_cases = [
        ("Full October", "2025-10-01", "2025-10-31"),
        ("First half", "2025-10-01", "2025-10-15"),
        ("Second half", "2025-10-16", "2025-10-31"),
    ]
    
    for i, (name, start, end) in enumerate(test_cases, 1):
        print(f"\n{i}. Test: {name} ({start} to {end})")
        result = tool.execute(start_date=start, end_date=end)
        
        if result.success:
            metrics = result.output['financial_metrics']
            print(f"   ✅ Analysis successful")
            print(f"      Gross Revenue: ${metrics['total_revenue']:,.2f}")
            print(f"      Total Fees: ${metrics['total_fees']:,.2f}")
            print(f"      Net Revenue: ${metrics['net_revenue']:,.2f}")
            print(f"      Profit Margin: {metrics['profit_margin_pct']:.2f}%")
        else:
            print(f"   ❌ Failed: {result.error}")
            return False
    
    return True


def test_comparison():
    """Test comparison tool."""
    print("\n" + "="*80)
    print("COMPARISON TEST")
    print("="*80)
    
    tool = ComparisonTool()
    
    test_cases = [
        ("Week 1 vs Week 2", "2025-10-01", "2025-10-07", "2025-10-08", "2025-10-14"),
        ("First half vs Second half", "2025-10-01", "2025-10-15", "2025-10-16", "2025-10-31"),
    ]
    
    for i, (name, p1_start, p1_end, p2_start, p2_end) in enumerate(test_cases, 1):
        print(f"\n{i}. Test: {name}")
        result = tool.execute(
            comparison_type='period',
            period1_start=p1_start,
            period1_end=p1_end,
            period2_start=p2_start,
            period2_end=p2_end
        )
        
        if result.success:
            comparison = result.output['comparison']
            print(f"   ✅ Comparison successful")
            print(f"      Period 1 Revenue: ${comparison['period1']['revenue']:,.2f}")
            print(f"      Period 2 Revenue: ${comparison['period2']['revenue']:,.2f}")
            print(f"      Revenue Growth: {comparison['growth']['revenue_pct']:.2f}%")
            print(f"      Order Growth: {comparison['growth']['orders_pct']:.2f}%")
        else:
            print(f"   ❌ Failed: {result.error}")
            return False
    
    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("ANALYSIS TOOLS INTEGRATION TEST SUITE")
    print("="*80)
    
    # Test all tools
    sales_ok = test_sales_trend()
    inventory_ok = test_inventory_analysis()
    product_ok = test_product_performance()
    financial_ok = test_financial_summary()
    comparison_ok = test_comparison()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Sales Trend: {'✅ PASSED' if sales_ok else '❌ FAILED'}")
    print(f"Inventory Analysis: {'✅ PASSED' if inventory_ok else '❌ FAILED'}")
    print(f"Product Performance: {'✅ PASSED' if product_ok else '❌ FAILED'}")
    print(f"Financial Summary: {'✅ PASSED' if financial_ok else '❌ FAILED'}")
    print(f"Comparison: {'✅ PASSED' if comparison_ok else '❌ FAILED'}")
    
    if all([sales_ok, inventory_ok, product_ok, financial_ok, comparison_ok]):
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Please review and fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())