from value_agent.metrics import analyze_value_case


def test_value_case_not_eligible_without_margin_of_safety():
    annuals = []
    for year in range(2015, 2025):
        annuals.append(
            {
                "fiscal_year": year,
                "revenue": 100 + (year - 2015) * 10,
                "net_income": 10 + (year - 2015),
                "operating_cash_flow": 11 + (year - 2015),
                "assets": 200 + (year - 2015) * 20,
                "liabilities": 50,
                "current_assets": 100,
                "current_liabilities": 50,
                "equity": 150 + (year - 2015) * 10,
                "long_term_debt": 20,
                "eps_diluted": 2.0 + (year - 2015) * 0.1,
                "shares_diluted": 10,
                "shares_outstanding": 10,
            }
        )
    result = analyze_value_case(
        annuals,
        price=1000,
        portfolio_value=10000,
        max_position_pct=0.05,
        first_tranche_pct=0.25,
    )
    assert result["decision"]["eligible"] is False
    assert result["allocation"]["minimum_units"] == 0


def test_value_case_eligible_with_conservative_price():
    annuals = []
    for year in range(2015, 2025):
        annuals.append(
            {
                "fiscal_year": year,
                "revenue": 100 + (year - 2015) * 10,
                "net_income": 10 + (year - 2015),
                "operating_cash_flow": 11 + (year - 2015),
                "assets": 200 + (year - 2015) * 20,
                "liabilities": 40,
                "current_assets": 100,
                "current_liabilities": 50,
                "equity": 150 + (year - 2015) * 10,
                "long_term_debt": 20,
                "eps_diluted": 4.0 + (year - 2015) * 0.1,
                "shares_diluted": 10,
                "shares_outstanding": 10,
            }
        )
    result = analyze_value_case(
        annuals,
        price=20,
        portfolio_value=10000,
        max_position_pct=0.05,
        first_tranche_pct=0.25,
    )
    assert result["decision"]["eligible"] is True
    assert result["allocation"]["minimum_units"] >= 1
