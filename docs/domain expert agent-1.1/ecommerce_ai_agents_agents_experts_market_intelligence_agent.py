--- ecommerce_ai_agents/agents/experts/market_intelligence_agent.py (原始)


+++ ecommerce_ai_agents/agents/experts/market_intelligence_agent.py (修改后)
"""
Global Market Intelligence Agent
作用：提供实时数据支持，包括汇率监控、竞品分析和季节性趋势预测
集成点：Data Analytics, Business Development
"""

import datetime
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ExchangeRate:
    """汇率数据"""
    base_currency: str
    target_currency: str
    rate: float
    change_24h: float  # 24小时变化百分比
    trend: str  # "up", "down", "stable"
    last_updated: datetime.datetime


@dataclass
class CompetitorInsight:
    """竞品洞察"""
    platform: str
    competitor_name: str
    product_category: str
    price_range: Dict[str, float]
    top_selling_skus: List[str]
    marketing_strategies: List[str]
    sentiment_score: float  # 0-100
    recent_activities: List[str]


@dataclass
class SeasonalTrend:
    """季节性趋势"""
    event_name: str
    target_markets: List[str]
    start_date: datetime.date
    end_date: datetime.date
    expected_growth_rate: float
    recommended_categories: List[str]
    strategic_suggestions: List[str]


@dataclass
class MacroRisk:
    """宏观风险预警"""
    risk_type: str
    affected_markets: List[str]
    severity: str  # "low", "medium", "high", "critical"
    description: str
    recommended_actions: List[str]
    detected_at: datetime.datetime


class GlobalMarketIntelligenceAgent:
    """
    全球市场情报 Agent

    功能:
    1. 实时汇率波动监控与定价建议
    2. 竞品在 TikTok/Amazon 的实时动态抓取与分析
    3. 季节性趋势预测（黑五、斋月、双11等）
    4. 宏观风险预警
    """

    def __init__(self):
        self.name = "Global Market Intelligence Agent"
        self.version = "1.0.0"

        # 模拟汇率数据源
        self.exchange_rates_db = {
            ("USD", "CNY"): 7.24,
            ("USD", "EUR"): 0.92,
            ("USD", "GBP"): 0.79,
            ("USD", "JPY"): 151.50,
            ("EUR", "CNY"): 7.87,
            ("GBP", "CNY"): 9.16,
        }

        # 季节性事件日历
        self.seasonal_events = {
            "black_friday": {
                "name": "Black Friday & Cyber Monday",
                "markets": ["US", "CA", "UK", "EU"],
                "month": 11,
                "categories": ["Electronics", "Fashion", "Home & Garden", "Toys"],
                "avg_growth": 0.45
            },
            "singles_day": {
                "name": "Singles' Day (Double 11)",
                "markets": ["CN", "SEA"],
                "month": 11,
                "categories": ["Beauty", "Fashion", "Electronics", "FMCG"],
                "avg_growth": 0.60
            },
            "ramadan": {
                "name": "Ramadan & Eid al-Fitr",
                "markets": ["ME", "ID", "MY", "PK"],
                "month": 3,  # 近似月份，实际每年变化
                "categories": ["Fashion", "Food & Beverage", "Home Decor", "Gifts"],
                "avg_growth": 0.35
            },
            "christmas": {
                "name": "Christmas Season",
                "markets": ["US", "CA", "UK", "EU", "AU"],
                "month": 12,
                "categories": ["Toys", "Electronics", "Fashion", "Gifts", "Home Decor"],
                "avg_growth": 0.40
            },
            "prime_day": {
                "name": "Amazon Prime Day",
                "markets": ["US", "UK", "EU", "JP"],
                "month": 7,
                "categories": ["Electronics", "Home", "Fashion", "Beauty"],
                "avg_growth": 0.30
            }
        }

        # 宏观风险规则
        self.risk_rules = [
            {
                "type": "trade_policy",
                "keywords": ["tariff", "sanction", "trade war"],
                "severity_map": {"tariff": "medium", "sanction": "critical", "trade war": "high"}
            },
            {
                "type": "currency_crisis",
                "keywords": ["devaluation", "inflation", "capital control"],
                "severity_map": {"devaluation": "high", "inflation": "medium", "capital control": "critical"}
            },
            {
                "type": "regulatory_change",
                "keywords": ["gdpr", "data privacy", "product safety", "compliance"],
                "severity_map": {"gdpr": "high", "data privacy": "high", "product safety": "medium"}
            }
        ]

    def get_exchange_rate(self, base: str, target: str) -> Optional[ExchangeRate]:
        """获取实时汇率"""
        key = (base.upper(), target.upper())
        if key in self.exchange_rates_db:
            base_rate = self.exchange_rates_db[key]
            # 模拟24小时波动 (-2% 到 +2%)
            change = random.uniform(-0.02, 0.02)
            current_rate = base_rate * (1 + change)

            trend = "stable"
            if change > 0.005:
                trend = "up"
            elif change < -0.005:
                trend = "down"

            return ExchangeRate(
                base_currency=base.upper(),
                target_currency=target.upper(),
                rate=round(current_rate, 4),
                change_24h=round(change * 100, 2),
                trend=trend,
                last_updated=datetime.datetime.now()
            )
        return None

    def analyze_pricing_strategy(self, base_price: float, base_currency: str,
                                target_currency: str, target_margin: float = 0.25) -> Dict[str, Any]:
        """
        基于汇率波动分析定价策略

        Args:
            base_price: 基础价格（本币）
            base_currency: 基础货币
            target_currency: 目标货币
            target_margin: 目标利润率

        Returns:
            定价建议报告
        """
        rate_info = self.get_exchange_rate(base_currency, target_currency)
        if not rate_info:
            return {"error": f"Unsupported currency pair: {base_currency}/{target_currency}"}

        # 计算基础外币价格
        base_foreign_price = base_price * rate_info.rate

        # 考虑汇率波动调整
        volatility_buffer = 0.02  # 2% 缓冲
        if rate_info.trend == "down" and abs(rate_info.change_24h) > 1.0:
            # 本币贬值，适当提价保护利润
            suggested_price = base_foreign_price * (1 + target_margin + volatility_buffer)
            recommendation = "Currency depreciating. Consider increasing price to maintain margins."
        elif rate_info.trend == "up" and abs(rate_info.change_24h) > 1.0:
            # 本币升值，可降价增强竞争力
            suggested_price = base_foreign_price * (1 + target_margin - volatility_buffer * 0.5)
            recommendation = "Currency appreciating. Consider reducing price to gain market share."
        else:
            suggested_price = base_foreign_price * (1 + target_margin)
            recommendation = "Stable exchange rate. Maintain standard pricing strategy."

        return {
            "base_price": {"amount": base_price, "currency": base_currency},
            "exchange_rate": rate_info.rate,
            "rate_trend": rate_info.trend,
            "rate_change_24h": f"{rate_info.change_24h}%",
            "calculated_foreign_price": round(base_foreign_price, 2),
            "suggested_retail_price": round(suggested_price, 2),
            "target_margin": f"{target_margin * 100}%",
            "recommendation": recommendation,
            "last_updated": rate_info.last_updated.isoformat()
        }

    def scan_competitors(self, platform: str, category: str,
                        region: str = "US") -> List[CompetitorInsight]:
        """
        扫描竞品动态

        Args:
            platform: 平台 (TikTok, Amazon, etc.)
            category: 产品类目
            region: 目标区域

        Returns:
            竞品洞察列表
        """
        # 模拟竞品数据
        mock_competitors = {
            "Electronics": [
                {
                    "name": "TechGiant Pro",
                    "price_range": {"min": 49.99, "max": 299.99},
                    "top_skus": ["TG-Phone-X", "TG-Buds-Pro", "TG-Watch-3"],
                    "strategies": ["Influencer partnerships", "Flash sales", "Bundle deals"],
                    "sentiment": 78,
                    "activities": ["Launched new earbuds", "20% off summer sale"]
                },
                {
                    "name": "ValueElectro",
                    "price_range": {"min": 19.99, "max": 149.99},
                    "top_skus": ["VE-Basic-Phone", "VE-Charger-Fast"],
                    "strategies": ["Low price leadership", "Free shipping"],
                    "sentiment": 65,
                    "activities": ["Expanded to EU market", "New warehouse opening"]
                }
            ],
            "Fashion": [
                {
                    "name": "StyleHub",
                    "price_range": {"min": 29.99, "max": 199.99},
                    "top_skus": ["SH-Summer-Dress", "SH-Denim-Jacket"],
                    "strategies": ["Sustainable fashion campaign", "User-generated content"],
                    "sentiment": 82,
                    "activities": ["Collaboration with eco-influencers", "New collection launch"]
                }
            ],
            "Home & Garden": [
                {
                    "name": "HomeComfort",
                    "price_range": {"min": 15.99, "max": 89.99},
                    "top_skus": ["HC-Smart-Lamp", "HC-Plant-Pot-Set"],
                    "strategies": ["DIY tutorials", "Home makeover contests"],
                    "sentiment": 71,
                    "activities": ["Viral TikTok video", "Limited edition release"]
                }
            ]
        }

        results = []
        competitors_data = mock_competitors.get(category, [])

        for comp in competitors_data:
            insight = CompetitorInsight(
                platform=platform,
                competitor_name=comp["name"],
                product_category=category,
                price_range=comp["price_range"],
                top_selling_skus=comp["top_skus"],
                marketing_strategies=comp["strategies"],
                sentiment_score=comp["sentiment"],
                recent_activities=comp["activities"]
            )
            results.append(insight)

        return results

    def get_seasonal_forecast(self, target_month: int = None,
                             target_markets: List[str] = None) -> List[SeasonalTrend]:
        """
        获取季节性趋势预测

        Args:
            target_month: 目标月份 (1-12)，默认当前月份
            target_markets: 目标市场列表

        Returns:
            季节性趋势列表
        """
        if target_month is None:
            target_month = datetime.datetime.now().month

        results = []

        for event_key, event_data in self.seasonal_events.items():
            # 检查月份匹配
            if event_data["month"] != target_month:
                continue

            # 检查市场匹配
            if target_markets:
                event_markets = set(event_data["markets"])
                target_set = set(target_markets)
                if not event_markets.intersection(target_set):
                    continue

            # 计算日期范围（简化处理）
            year = datetime.datetime.now().year
            if event_data["month"] < target_month or (event_data["month"] == target_month and datetime.datetime.now().day > 15):
                year += 1

            start_date = datetime.date(year, event_data["month"], 1)
            if event_data["month"] == 12:
                end_date = datetime.date(year, 12, 31)
            else:
                end_date = datetime.date(year, event_data["month"] + 1, 7)

            # 生成战略建议
            suggestions = [
                f"Start inventory preparation 8-12 weeks before {event_data['name']}",
                f"Focus marketing budget on {', '.join(event_data['categories'][:3])}",
                "Prepare customer support team for increased volume",
                "Consider early-bird promotions 2 weeks before the event"
            ]

            trend = SeasonalTrend(
                event_name=event_data["name"],
                target_markets=event_data["markets"],
                start_date=start_date,
                end_date=end_date,
                expected_growth_rate=event_data["avg_growth"],
                recommended_categories=event_data["categories"],
                strategic_suggestions=suggestions
            )
            results.append(trend)

        return results

    def assess_macro_risks(self, news_keywords: List[str],
                          affected_regions: List[str] = None) -> List[MacroRisk]:
        """
        评估宏观风险

        Args:
            news_keywords: 新闻关键词列表
            affected_regions: 受影响区域

        Returns:
            风险预警列表
        """
        results = []

        for rule in self.risk_rules:
            matched_keywords = []
            max_severity = None

            for keyword in news_keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in rule["keywords"]:
                    matched_keywords.append(keyword)
                    severity = rule["severity_map"].get(keyword_lower, "medium")

                    # 更新最高严重程度
                    severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                    if max_severity is None or severity_order.get(severity, 0) > severity_order.get(max_severity, 0):
                        max_severity = severity

            if matched_keywords:
                risk = MacroRisk(
                    risk_type=rule["type"],
                    affected_markets=affected_regions or ["Global"],
                    severity=max_severity or "medium",
                    description=f"Detected potential {rule['type']} risks related to: {', '.join(matched_keywords)}",
                    recommended_actions=self._get_risk_actions(rule["type"], max_severity),
                    detected_at=datetime.datetime.now()
                )
                results.append(risk)

        return results

    def _get_risk_actions(self, risk_type: str, severity: str) -> List[str]:
        """根据风险类型和严重程度生成建议行动"""
        actions = {
            "trade_policy": [
                "Review current tariff classifications",
                "Consider alternative sourcing locations",
                "Adjust pricing strategy to absorb potential tariff increases",
                "Consult with trade compliance experts"
            ],
            "currency_crisis": [
                "Implement dynamic pricing adjustments",
                "Consider hedging strategies",
                "Diversify revenue currencies",
                "Monitor cash flow closely"
            ],
            "regulatory_change": [
                "Conduct compliance audit",
                "Update product documentation",
                "Train customer support on new regulations",
                "Review data handling practices"
            ]
        }

        base_actions = actions.get(risk_type, ["Monitor situation closely", "Consult legal advisors"])

        if severity in ["high", "critical"]:
            base_actions.insert(0, "URGENT: Form crisis management team")
            base_actions.insert(1, "Prepare contingency plans immediately")

        return base_actions

    def generate_intelligence_report(self, target_market: str,
                                    product_category: str,
                                    base_currency: str = "CNY",
                                    target_currency: str = "USD") -> Dict[str, Any]:
        """
        生成综合情报报告

        Args:
            target_market: 目标市场
            product_category: 产品类目
            base_currency: 基础货币
            target_currency: 目标货币

        Returns:
            完整情报报告
        """
        current_month = datetime.datetime.now().month

        # 1. 汇率分析
        pricing_analysis = self.analyze_pricing_strategy(
            base_price=100.0,  # 示例基础价格
            base_currency=base_currency,
            target_currency=target_currency
        )

        # 2. 竞品分析
        competitors = self.scan_competitors(
            platform="Amazon",  # 默认平台
            category=product_category,
            region=target_market
        )

        # 3. 季节性趋势
        seasonal_trends = self.get_seasonal_forecast(
            target_month=current_month,
            target_markets=[target_market]
        )

        # 4. 风险评估（模拟一些关键词）
        macro_risks = self.assess_macro_risks(
            news_keywords=["tariff", "data privacy"],
            affected_regions=[target_market]
        )

        report = {
            "report_metadata": {
                "generated_at": datetime.datetime.now().isoformat(),
                "target_market": target_market,
                "product_category": product_category,
                "agent_version": self.version
            },
            "currency_analysis": pricing_analysis,
            "competitive_landscape": [asdict(c) for c in competitors],
            "seasonal_opportunities": [asdict(s) for s in seasonal_trends],
            "risk_alerts": [asdict(r) for r in macro_risks],
            "executive_summary": self._generate_executive_summary(
                pricing_analysis, competitors, seasonal_trends, macro_risks
            )
        }

        return report

    def _generate_executive_summary(self, pricing: Dict, competitors: List[CompetitorInsight],
                                   trends: List[SeasonalTrend], risks: List[MacroRisk]) -> str:
        """生成执行摘要"""
        summary_parts = []

        # 汇率部分
        if "recommendation" in pricing:
            summary_parts.append(f"Pricing: {pricing['recommendation']}")

        # 竞争部分
        if competitors:
            top_competitor = competitors[0]
            summary_parts.append(f"Competition: {top_competitor.competitor_name} leads with {len(top_competitor.top_selling_skus)} top SKUs")

        # 季节部分
        if trends:
            summary_parts.append(f"Opportunity: {trends[0].event_name} approaching with {trends[0].expected_growth_rate*100:.0f}% expected growth")

        # 风险部分
        if risks:
            high_risks = [r for r in risks if r.severity in ["high", "critical"]]
            if high_risks:
                summary_parts.append(f"⚠️ Alert: {len(high_risks)} high-severity risk(s) detected requiring immediate attention")

        return " | ".join(summary_parts) if summary_parts else "No significant intelligence updates."


# 使用示例
if __name__ == "__main__":
    agent = GlobalMarketIntelligenceAgent()

    print("=" * 60)
    print("Global Market Intelligence Agent Demo")
    print("=" * 60)

    # 1. 汇率与定价分析
    print("\n1. Currency & Pricing Analysis:")
    pricing = agent.analyze_pricing_strategy(
        base_price=150.0,
        base_currency="CNY",
        target_currency="USD",
        target_margin=0.30
    )
    print(f"Suggested USD Price: ${pricing['suggested_retail_price']}")
    print(f"Recommendation: {pricing['recommendation']}")

    # 2. 竞品分析
    print("\n2. Competitive Intelligence:")
    competitors = agent.scan_competitors("Amazon", "Electronics", "US")
    for comp in competitors:
        print(f"- {comp.competitor_name}: Sentiment {comp.sentiment_score}/100")
        print(f"  Top SKUs: {', '.join(comp.top_selling_skus)}")

    # 3. 季节性趋势
    print("\n3. Seasonal Trends:")
    trends = agent.get_seasonal_forecast(target_month=11)
    for trend in trends:
        print(f"- {trend.event_name}: {trend.expected_growth_rate*100:.0f}% growth expected")
        print(f"  Focus categories: {', '.join(trend.recommended_categories[:3])}")

    # 4. 综合报告
    print("\n4. Full Intelligence Report:")
    report = agent.generate_intelligence_report(
        target_market="US",
        product_category="Electronics",
        base_currency="CNY",
        target_currency="USD"
    )
    print(f"Executive Summary: {report['executive_summary']}")