--- ecommerce_ai_agents/agents/experts/supply_chain_logistics_agent.py (原始)


+++ ecommerce_ai_agents/agents/experts/supply_chain_logistics_agent.py (修改后)
"""
Supply Chain & Logistics Agent for Cross-Border E-commerce
供应链与物流智能 Agent

This agent solves critical logistics timing and cost challenges in cross-border e-commerce.
Features:
- Automatic optimal logistics routing (Air vs Sea vs Overseas Warehouse)
- Real-time tariff policy monitoring and Landed Cost estimation
- Inventory turnover prediction and auto-replenishment suggestions
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransportMode(Enum):
    """运输方式"""
    AIR_FREIGHT = "air_freight"
    SEA_FREIGHT = "sea_freight"
    RAIL_FREIGHT = "rail_freight"
    EXPRESS = "express"
    OVERSEAS_WAREHOUSE = "overseas_warehouse"


class Incoterm(Enum):
    """国际贸易术语"""
    EXW = "EXW"
    FOB = "FOB"
    CIF = "CIF"
    DDP = "DDP"
    DAP = "DAP"


@dataclass
class LogisticsOption:
    """物流方案选项"""
    mode: TransportMode
    provider: str
    transit_days: int
    cost_per_kg: float
    min_weight_kg: float
    reliability_score: float  # 0-10
    tracking_available: bool
    description: str


@dataclass
class LandedCostBreakdown:
    """落地成本明细"""
    product_cost: float
    shipping_cost: float
    insurance_cost: float
    customs_duty: float
    vat_gst: float
    handling_fees: float
    total_landed_cost: float
    currency: str = "USD"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InventoryPrediction:
    """库存预测结果"""
    product_sku: str
    current_stock: int
    predicted_daily_sales: float
    days_of_stock_remaining: int
    reorder_point: int
    suggested_order_quantity: int
    recommended_order_date: str
    safety_stock_days: int = 15
    lead_time_days: int = 0


class SupplyChainLogisticsAgent:
    """
    供应链与物流智能 Agent

    核心功能：
    1. 根据目的地自动计算最优物流方案（空运 vs 海运 vs 海外仓）
    2. 实时监控关税政策变化，预估落地成本 (Landed Cost)
    3. 预测库存周转，自动触发补货建议
    """

    def __init__(self):
        self.name = "SupplyChainLogisticsAgent"
        self.description = "AI Agent for optimizing cross-border logistics, tariff calculation, and inventory management"

        # 物流商数据库
        self.logistics_providers = self._init_logistics_providers()

        # 关税税率库 (简化版，实际应连接实时API)
        self.tariff_rates = self._init_tariff_rates()

        # 增值税/消费税库
        self.vat_rates = self._init_vat_rates()

        # 海外仓位置
        self.overseas_warehouses = self._init_overseas_warehouses()

    def _init_logistics_providers(self) -> Dict[str, List[LogisticsOption]]:
        """初始化物流商数据"""
        return {
            "CN_to_US": [
                LogisticsOption(
                    mode=TransportMode.AIR_FREIGHT,
                    provider="DHL Express",
                    transit_days=5,
                    cost_per_kg=8.5,
                    min_weight_kg=0.5,
                    reliability_score=9.2,
                    tracking_available=True,
                    description="Fast air freight with full tracking"
                ),
                LogisticsOption(
                    mode=TransportMode.SEA_FREIGHT,
                    provider="COSCO Shipping",
                    transit_days=25,
                    cost_per_kg=2.3,
                    min_weight_kg=100,
                    reliability_score=8.5,
                    tracking_available=True,
                    description="Economical sea freight for bulk shipments"
                ),
                LogisticsOption(
                    mode=TransportMode.EXPRESS,
                    provider="FedEx International",
                    transit_days=3,
                    cost_per_kg=12.0,
                    min_weight_kg=0.5,
                    reliability_score=9.5,
                    tracking_available=True,
                    description="Premium express delivery"
                ),
                LogisticsOption(
                    mode=TransportMode.OVERSEAS_WAREHOUSE,
                    provider="US West Coast Warehouse",
                    transit_days=2,
                    cost_per_kg=1.5,
                    min_weight_kg=1,
                    reliability_score=9.0,
                    tracking_available=True,
                    description="Local delivery from overseas warehouse"
                )
            ],
            "CN_to_EU": [
                LogisticsOption(
                    mode=TransportMode.AIR_FREIGHT,
                    provider="Lufthansa Cargo",
                    transit_days=6,
                    cost_per_kg=9.0,
                    min_weight_kg=0.5,
                    reliability_score=9.0,
                    tracking_available=True,
                    description="Reliable air freight to Europe"
                ),
                LogisticsOption(
                    mode=TransportMode.RAIL_FREIGHT,
                    provider="China-Europe Railway Express",
                    transit_days=18,
                    cost_per_kg=4.5,
                    min_weight_kg=50,
                    reliability_score=8.3,
                    tracking_available=True,
                    description="Cost-effective rail transport via Belt and Road"
                ),
                LogisticsOption(
                    mode=TransportMode.SEA_FREIGHT,
                    provider="Maersk Line",
                    transit_days=35,
                    cost_per_kg=1.8,
                    min_weight_kg=100,
                    reliability_score=8.0,
                    tracking_available=True,
                    description="Most economical sea route to EU"
                )
            ],
            "CN_to_JP": [
                LogisticsOption(
                    mode=TransportMode.EXPRESS,
                    provider="SF International",
                    transit_days=2,
                    cost_per_kg=6.5,
                    min_weight_kg=0.5,
                    reliability_score=9.3,
                    tracking_available=True,
                    description="Fast express to Japan"
                ),
                LogisticsOption(
                    mode=TransportMode.SEA_FREIGHT,
                    provider="NYK Line",
                    transit_days=7,
                    cost_per_kg=1.5,
                    min_weight_kg=50,
                    reliability_score=8.7,
                    tracking_available=True,
                    description="Quick sea route to Japan"
                )
            ],
            "CN_to_SEA": [
                LogisticsOption(
                    mode=TransportMode.EXPRESS,
                    provider="J&T Express",
                    transit_days=3,
                    cost_per_kg=5.0,
                    min_weight_kg=0.5,
                    reliability_score=8.8,
                    tracking_available=True,
                    description="Popular express for Southeast Asia"
                ),
                LogisticsOption(
                    mode=TransportMode.SEA_FREIGHT,
                    provider="Wan Hai Lines",
                    transit_days=10,
                    cost_per_kg=1.2,
                    min_weight_kg=50,
                    reliability_score=8.2,
                    tracking_available=True,
                    description="Affordable sea freight to SEA"
                )
            ]
        }

    def _init_tariff_rates(self) -> Dict[str, Dict[str, float]]:
        """
        初始化关税税率库
        格式：{目标国家: {HS编码前缀: 税率}}
        注意：实际生产环境应连接海关API获取实时税率
        """
        return {
            "US": {
                "61": 0.165,  # 针织服装
                "62": 0.160,  # 梭织服装
                "64": 0.090,  # 鞋类
                "85": 0.034,  # 电子产品
                "95": 0.000,  # 玩具
                "default": 0.050
            },
            "EU": {
                "61": 0.120,
                "62": 0.120,
                "64": 0.080,
                "85": 0.000,
                "95": 0.047,
                "default": 0.045
            },
            "JP": {
                "61": 0.096,
                "62": 0.096,
                "64": 0.030,
                "85": 0.000,
                "95": 0.000,
                "default": 0.030
            },
            "UK": {
                "61": 0.120,
                "62": 0.120,
                "64": 0.080,
                "85": 0.000,
                "95": 0.047,
                "default": 0.040
            },
            "AU": {
                "61": 0.050,
                "62": 0.050,
                "64": 0.050,
                "85": 0.000,
                "95": 0.000,
                "default": 0.050
            },
            "SEA": {
                "61": 0.100,
                "62": 0.100,
                "64": 0.080,
                "85": 0.050,
                "95": 0.050,
                "default": 0.070
            }
        }

    def _init_vat_rates(self) -> Dict[str, float]:
        """初始化增值税/消费税率"""
        return {
            "US": 0.0,  # 美国无联邦增值税，州税另计
            "EU": 0.20,  # 欧盟平均VAT
            "DE": 0.19,
            "FR": 0.20,
            "IT": 0.22,
            "ES": 0.21,
            "UK": 0.20,
            "JP": 0.10,
            "AU": 0.10,
            "CA": 0.05,
            "SG": 0.08,
            "TH": 0.07,
            "VN": 0.10,
            "MY": 0.00,  # 马来西亚暂无GST
            "ID": 0.11,
            "PH": 0.12
        }

    def _init_overseas_warehouses(self) -> Dict[str, Dict]:
        """初始化海外仓信息"""
        return {
            "US_WEST": {
                "location": "Los Angeles, CA",
                "coverage": ["US", "CA", "MX"],
                "storage_cost_per_cbm_month": 15.0,
                "handling_fee_per_order": 3.5,
                "last_mile_cost_per_kg": 1.2
            },
            "US_EAST": {
                "location": "New York, NJ",
                "coverage": ["US", "CA"],
                "storage_cost_per_cbm_month": 18.0,
                "handling_fee_per_order": 4.0,
                "last_mile_cost_per_kg": 1.5
            },
            "EU_CENTRAL": {
                "location": "Hamburg, Germany",
                "coverage": ["DE", "FR", "NL", "BE", "PL"],
                "storage_cost_per_cbm_month": 20.0,
                "handling_fee_per_order": 4.5,
                "last_mile_cost_per_kg": 2.0
            },
            "EU_SOUTH": {
                "location": "Barcelona, Spain",
                "coverage": ["ES", "IT", "PT"],
                "storage_cost_per_cbm_month": 17.0,
                "handling_fee_per_order": 4.0,
                "last_mile_cost_per_kg": 1.8
            },
            "UK": {
                "location": "London, UK",
                "coverage": ["UK"],
                "storage_cost_per_cbm_month": 22.0,
                "handling_fee_per_order": 4.5,
                "last_mile_cost_per_kg": 2.2
            },
            "JP": {
                "location": "Tokyo, Japan",
                "coverage": ["JP"],
                "storage_cost_per_cbm_month": 25.0,
                "handling_fee_per_order": 5.0,
                "last_mile_cost_per_kg": 2.5
            },
            "SEA_HUB": {
                "location": "Singapore",
                "coverage": ["SG", "MY", "TH", "ID"],
                "storage_cost_per_cbm_month": 18.0,
                "handling_fee_per_order": 3.0,
                "last_mile_cost_per_kg": 1.5
            }
        }

    def calculate_optimal_logistics(
        self,
        origin_country: str,
        destination_country: str,
        weight_kg: float,
        volume_cbm: float,
        product_value_usd: float,
        urgency: str = "normal",
        hs_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        根据目的地自动计算最优物流方案

        Args:
            origin_country: 出发国家代码 (e.g., "CN")
            destination_country: 目的国家代码 (e.g., "US", "DE", "JP")
            weight_kg: 货物重量 (kg)
            volume_cbm: 货物体积 (立方米)
            product_value_usd: 货物价值 (USD)
            urgency: 紧急程度 ("urgent", "normal", "economical")
            hs_code: HS编码前缀 (用于关税计算)

        Returns:
            包含推荐方案和备选方案的字典
        """
        route_key = f"{origin_country}_to_{destination_country}"

        # 标准化目的地代码
        dest_region = self._get_region_code(destination_country)
        if route_key not in self.logistics_providers:
            route_key = f"{origin_country}_to_{dest_region}"

        if route_key not in self.logistics_providers:
            return {
                "success": False,
                "error": f"No logistics data available for route: {route_key}",
                "suggestion": "Please contact our support team for custom logistics solutions"
            }

        available_options = self.logistics_providers[route_key]

        # 过滤不符合重量要求的选项
        valid_options = [
            opt for opt in available_options
            if weight_kg >= opt.min_weight_kg
        ]

        if not valid_options:
            return {
                "success": False,
                "error": f"No logistics options available for weight: {weight_kg}kg",
                "min_weight_required": min(opt.min_weight_kg for opt in available_options)
            }

        # 根据紧急程度评分
        scored_options = []
        for option in valid_options:
            score = self._calculate_option_score(
                option, urgency, weight_kg, volume_cbm, product_value_usd
            )
            scored_options.append({
                "option": option,
                "score": score,
                "total_cost": option.cost_per_kg * weight_kg,
                "estimated_delivery_days": option.transit_days
            })

        # 按分数排序
        scored_options.sort(key=lambda x: x["score"], reverse=True)

        # 计算落地成本
        landed_costs = []
        for item in scored_options:
            landed_cost = self.calculate_landed_cost(
                destination_country=destination_country,
                product_value=product_value_usd,
                weight_kg=weight_kg,
                shipping_cost=item["total_cost"],
                hs_code=hs_code,
                incoterm=Incoterm.DDP
            )
            item["landed_cost"] = landed_cost
            landed_costs.append(item)

        recommendation = landed_costs[0] if landed_costs else None

        return {
            "success": True,
            "route": f"{origin_country} → {destination_country}",
            "recommendation": {
                "mode": recommendation["option"].mode.value if recommendation else None,
                "provider": recommendation["option"].provider if recommendation else None,
                "transit_days": recommendation["estimated_delivery_days"] if recommendation else None,
                "shipping_cost_usd": round(recommendation["total_cost"], 2) if recommendation else None,
                "total_landed_cost_usd": round(recommendation["landed_cost"].total_landed_cost, 2) if recommendation else None,
                "score": round(recommendation["score"], 2) if recommendation else None,
                "reason": self._generate_recommendation_reason(recommendation, urgency) if recommendation else None
            },
            "alternatives": [
                {
                    "mode": item["option"].mode.value,
                    "provider": item["option"].provider,
                    "transit_days": item["estimated_delivery_days"],
                    "shipping_cost_usd": round(item["total_cost"], 2),
                    "total_landed_cost_usd": round(item["landed_cost"].total_landed_cost, 2),
                    "score": round(item["score"], 2)
                }
                for item in landed_costs[1:4]  # 返回前3个备选方案
            ],
            "calculation_timestamp": datetime.now().isoformat()
        }

    def _get_region_code(self, country_code: str) -> str:
        """将国家代码映射到区域代码"""
        eu_countries = ["DE", "FR", "IT", "ES", "NL", "BE", "PL", "AT", "SE", "DK"]
        sea_countries = ["SG", "MY", "TH", "VN", "ID", "PH"]

        if country_code in eu_countries:
            return "EU"
        elif country_code in sea_countries:
            return "SEA"
        else:
            return country_code

    def _calculate_option_score(
        self,
        option: LogisticsOption,
        urgency: str,
        weight_kg: float,
        volume_cbm: float,
        product_value_usd: float
    ) -> float:
        """
        计算物流方案综合得分
        考虑因素：时效、成本、可靠性、货物价值
        """
        base_score = option.reliability_score * 10  # 基础分 (0-100)

        # 紧急程度权重
        urgency_weights = {
            "urgent": {"time": 0.5, "cost": 0.2, "reliability": 0.3},
            "normal": {"time": 0.3, "cost": 0.4, "reliability": 0.3},
            "economical": {"time": 0.1, "cost": 0.6, "reliability": 0.3}
        }
        weights = urgency_weights.get(urgency, urgency_weights["normal"])

        # 时效得分 (越快越好)
        time_score = max(0, 100 - option.transit_days * 3)

        # 成本得分 (越低越好)
        total_cost = option.cost_per_kg * weight_kg
        cost_score = max(0, 100 - (total_cost / max(weight_kg * 5, 1)) * 10)

        # 加权总分
        final_score = (
            time_score * weights["time"] +
            cost_score * weights["cost"] +
            base_score * weights["reliability"]
        )

        # 高价值货物增加可靠性权重
        if product_value_usd > 1000:
            final_score += option.reliability_score * 5

        return final_score

    def _generate_recommendation_reason(
        self,
        recommendation: Dict,
        urgency: str
    ) -> str:
        """生成推荐理由"""
        option = recommendation["option"]
        reasons = []

        if urgency == "urgent":
            reasons.append(f"最快 {option.transit_days} 天送达")
        elif urgency == "economical":
            reasons.append(f"最具成本效益，每公斤 ${option.cost_per_kg}")
        else:
            reasons.append(f"平衡时效 ({option.transit_days}天) 与成本 (${option.cost_per_kg}/kg)")

        reasons.append(f"可靠性评分：{option.reliability_score}/10")

        if option.tracking_available:
            reasons.append("支持全程追踪")

        if option.mode == TransportMode.OVERSEAS_WAREHOUSE:
            reasons.append("从海外仓发货，末端配送更快")

        return "; ".join(reasons)

    def calculate_landed_cost(
        self,
        destination_country: str,
        product_value: float,
        weight_kg: float,
        shipping_cost: float,
        hs_code: Optional[str] = None,
        incoterm: Incoterm = Incoterm.DDP,
        insurance_rate: float = 0.003
    ) -> LandedCostBreakdown:
        """
        计算落地成本 (Landed Cost)

        Args:
            destination_country: 目的国
            product_value: 产品价值 (USD)
            weight_kg: 重量 (kg)
            shipping_cost: 运费 (USD)
            hs_code: HS编码前缀
            incoterm: 贸易术语
            insurance_rate: 保险费率

        Returns:
            LandedCostBreakdown对象
        """
        # 保险费
        insurance_cost = product_value * insurance_rate

        # 关税计算
        duty_rate = self._get_duty_rate(destination_country, hs_code)
        customs_duty = product_value * duty_rate

        # 增值税计算 (基于 CIF 价值 + 关税)
        cif_value = product_value + shipping_cost + insurance_cost
        vat_rate = self.vat_rates.get(destination_country, self.vat_rates.get("EU", 0.20))

        # 某些国家对进口商品免征增值税门槛
        vat_thresholds = {
            "US": 800,  # de minimis threshold
            "EU": 150,
            "UK": 135,
            "AU": 1000,
            "JP": 10000  # JPY
        }

        vat_taxable = cif_value + customs_duty
        if destination_country in vat_thresholds:
            if cif_value < vat_thresholds[destination_country]:
                vat_rate = 0  # 低于阈值免增值税

        vat_gst = vat_taxable * vat_rate

        # 处理费 (估算)
        handling_fees = 15.0 + (weight_kg * 0.5)

        # 总落地成本
        total_landed_cost = (
            product_value +
            shipping_cost +
            insurance_cost +
            customs_duty +
            vat_gst +
            handling_fees
        )

        return LandedCostBreakdown(
            product_cost=round(product_value, 2),
            shipping_cost=round(shipping_cost, 2),
            insurance_cost=round(insurance_cost, 2),
            customs_duty=round(customs_duty, 2),
            vat_gst=round(vat_gst, 2),
            handling_fees=round(handling_fees, 2),
            total_landed_cost=round(total_landed_cost, 2),
            currency="USD"
        )

    def _get_duty_rate(self, country_code: str, hs_code: Optional[str]) -> float:
        """获取关税率"""
        region = self._get_region_code(country_code)

        if region not in self.tariff_rates:
            return self.tariff_rates.get("US", {}).get("default", 0.05)

        if hs_code and len(hs_code) >= 2:
            hs_prefix = hs_code[:2]
            rate = self.tariff_rates[region].get(hs_prefix)
            if rate is not None:
                return rate

        return self.tariff_rates[region].get("default", 0.05)

    def predict_inventory_and_replenish(
        self,
        product_sku: str,
        current_stock: int,
        sales_history: List[Dict[str, Any]],
        lead_time_days: int = 15,
        safety_stock_days: int = 15,
        upcoming_promotions: Optional[List[Dict]] = None
    ) -> InventoryPrediction:
        """
        预测库存周转并生成补货建议

        Args:
            product_sku: 产品SKU
            current_stock: 当前库存数量
            sales_history: 销售历史数据 (过去30-90天)
            lead_time_days: 补货提前期 (天)
            safety_stock_days: 安全库存天数
            upcoming_promotions: 即将到来的促销活动

        Returns:
            InventoryPrediction对象
        """
        if not sales_history:
            return InventoryPrediction(
                product_sku=product_sku,
                current_stock=current_stock,
                predicted_daily_sales=0,
                days_of_stock_remaining=0,
                reorder_point=0,
                suggested_order_quantity=0,
                recommended_order_date=datetime.now().strftime("%Y-%m-%d"),
                safety_stock_days=safety_stock_days,
                lead_time_days=lead_time_days
            )

        # 计算加权平均日销量 (最近的数据权重更高)
        weighted_sales = []
        total_weight = 0

        for i, sale in enumerate(sales_history[-30:]):  # 取最近30天
            weight = 1 + (i * 0.1)  # 越近的数据权重越高
            daily_sales = sale.get("quantity_sold", 0)
            weighted_sales.append(daily_sales * weight)
            total_weight += weight

        avg_daily_sales = sum(weighted_sales) / total_weight if total_weight > 0 else 0

        # 考虑促销活动影响
        if upcoming_promotions:
            today = datetime.now()
            for promo in upcoming_promotions:
                promo_start = datetime.fromisoformat(promo.get("start_date", ""))
                promo_end = datetime.fromisoformat(promo.get("end_date", ""))
                if promo_start <= today <= promo_end:
                    uplift_factor = promo.get("sales_uplift_factor", 1.5)
                    avg_daily_sales *= uplift_factor
                    break

        # 计算库存可支撑天数
        days_remaining = current_stock / avg_daily_sales if avg_daily_sales > 0 else 999

        # 计算再订货点
        reorder_point = int(avg_daily_sales * (lead_time_days + safety_stock_days))

        # 计算建议订单量 (EOQ简化版)
        if days_remaining < (lead_time_days + safety_stock_days):
            target_stock = avg_daily_sales * (lead_time_days + safety_stock_days + 30)  # 补充到覆盖未来30+LT+SS
            suggested_quantity = int(target_stock - current_stock)
            suggested_quantity = max(suggested_quantity, 0)
        else:
            suggested_quantity = 0

        # 计算建议下单日期
        if suggested_quantity > 0:
            days_until_order = days_remaining - lead_time_days - safety_stock_days
            order_date = datetime.now() + timedelta(days=max(0, days_until_order))
        else:
            order_date = datetime.now()

        return InventoryPrediction(
            product_sku=product_sku,
            current_stock=current_stock,
            predicted_daily_sales=round(avg_daily_sales, 2),
            days_of_stock_remaining=int(days_remaining),
            reorder_point=reorder_point,
            suggested_order_quantity=suggested_quantity,
            recommended_order_date=order_date.strftime("%Y-%m-%d"),
            safety_stock_days=safety_stock_days,
            lead_time_days=lead_time_days
        )

    def monitor_tariff_changes(
        self,
        country_codes: List[str],
        hs_codes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        监控关税政策变化
        注意：实际生产环境应连接海关API或第三方数据服务

        Args:
            country_codes: 要监控的国家代码列表
            hs_codes: 要监控的HS编码列表 (可选)

        Returns:
            关税政策更新摘要
        """
        # 模拟数据 - 实际应调用外部API
        recent_changes = []

        for country in country_codes:
            region = self._get_region_code(country)
            base_rate = self.tariff_rates.get(region, {}).get("default", 0.05)

            # 模拟一些变化
            if country == "US":
                recent_changes.append({
                    "country": "US",
                    "effective_date": "2024-01-15",
                    "change_type": "adjustment",
                    "description": "Section 301 tariffs on certain Chinese goods adjusted",
                    "affected_hs_codes": ["8517", "8528", "9403"],
                    "old_rate": 0.25,
                    "new_rate": 0.20,
                    "impact": "Reduced duty burden for electronics and furniture"
                })

            if country in ["DE", "FR", "IT"]:
                recent_changes.append({
                    "country": country,
                    "effective_date": "2024-03-01",
                    "change_type": "regulation",
                    "description": "EU Carbon Border Adjustment Mechanism (CBAM) reporting requirements",
                    "affected_hs_codes": ["72", "76", "28"],
                    "note": "New reporting obligations for steel, aluminum, and chemicals",
                    "impact": "Additional compliance costs for affected categories"
                })

        return {
            "monitoring_timestamp": datetime.now().isoformat(),
            "countries_monitored": country_codes,
            "total_changes_found": len(recent_changes),
            "recent_changes": recent_changes,
            "recommendations": [
                "Review product classifications for affected HS codes",
                "Update landed cost calculations with new rates",
                "Consider supply chain diversification if tariffs increase significantly"
            ] if recent_changes else ["No significant tariff changes detected"]
        }

    def integrate_with_sales_performance(
        self,
        product_data: List[Dict],
        target_markets: List[str]
    ) -> Dict[str, Any]:
        """
        与销售绩效模块集成
        提供物流成本对利润率的影响分析
        """
        analysis_results = []

        for product in product_data:
            sku = product.get("sku")
            product_value = product.get("price_usd", 0)
            weight = product.get("weight_kg", 1)
            hs_code = product.get("hs_code")

            market_analysis = {}
            for market in target_markets:
                logistics_result = self.calculate_optimal_logistics(
                    origin_country="CN",
                    destination_country=market,
                    weight_kg=weight,
                    volume_cbm=weight * 0.005,  # 估算体积
                    product_value_usd=product_value,
                    urgency="normal",
                    hs_code=hs_code
                )

                if logistics_result["success"]:
                    landed_cost = logistics_result["recommendation"]["total_landed_cost_usd"]
                    profit_margin = ((product_value - landed_cost) / product_value * 100) if product_value > 0 else 0

                    market_analysis[market] = {
                        "landed_cost_usd": landed_cost,
                        "profit_margin_percent": round(profit_margin, 2),
                        "recommended_shipping_mode": logistics_result["recommendation"]["mode"],
                        "delivery_days": logistics_result["recommendation"]["transit_days"]
                    }

            analysis_results.append({
                "sku": sku,
                "market_analysis": market_analysis
            })

        return {
            "analysis_type": "logistics_impact_on_profitability",
            "products_analyzed": len(analysis_results),
            "results": analysis_results,
            "timestamp": datetime.now().isoformat()
        }

    def integrate_with_business_development(
        self,
        target_country: str,
        product_categories: List[str]
    ) -> Dict[str, Any]:
        """
        与商务拓展模块集成
        评估新市场进入的物流可行性
        """
        feasibility_report = {
            "target_market": target_country,
            "evaluation_timestamp": datetime.now().isoformat(),
            "logistics_readiness": {},
            "recommendations": []
        }

        # 评估物流基础设施
        route_key = f"CN_to_{self._get_region_code(target_country)}"
        has_direct_routes = route_key in self.logistics_providers

        feasibility_report["logistics_readiness"] = {
            "direct_shipping_available": has_direct_routes,
            "overseas_warehouse_nearby": any(
                target_country in wh["coverage"]
                for wh in self.overseas_warehouses.values()
            ),
            "average_transit_time_days": self._estimate_avg_transit_time(target_country),
            "cost_competitiveness": self._assess_cost_competitiveness(target_country)
        }

        # 生成市场进入建议
        if has_direct_routes:
            feasibility_report["recommendations"].append(
                f"Direct logistics routes available to {target_country}. Consider starting with air freight for testing."
            )
        else:
            feasibility_report["recommendations"].append(
                f"No direct routes to {target_country}. Consider transshipment via regional hub or overseas warehouse."
            )

        # 检查关税壁垒
        avg_duty_rate = sum(
            self.tariff_rates.get(self._get_region_code(target_country), {}).get(hs[:2], 0.05)
            for hs in ["61", "62", "64", "85", "95"]
        ) / 5

        if avg_duty_rate > 0.15:
            feasibility_report["recommendations"].append(
                f"High average tariff rate ({avg_duty_rate:.1%}). Consider local assembly or FTZ strategies."
            )

        return feasibility_report

    def _estimate_avg_transit_time(self, country_code: str) -> int:
        """估算平均运输时间"""
        region = self._get_region_code(country_code)
        route_key = f"CN_to_{region}"

        if route_key in self.logistics_providers:
            options = self.logistics_providers[route_key]
            return int(sum(opt.transit_days for opt in options) / len(options))

        return 30  # 默认值

    def _assess_cost_competitiveness(self, country_code: str) -> str:
        """评估成本竞争力"""
        region = self._get_region_code(country_code)
        route_key = f"CN_to_{region}"

        if route_key in self.logistics_providers:
            options = self.logistics_providers[route_key]
            avg_cost = sum(opt.cost_per_kg for opt in options) / len(options)

            if avg_cost < 3.0:
                return "Highly Competitive"
            elif avg_cost < 6.0:
                return "Competitive"
            elif avg_cost < 10.0:
                return "Moderate"
            else:
                return "Challenging"

        return "Unknown"

    def integrate_with_event_scheduler(
        self,
        event_name: str,
        event_date: str,
        expected_order_volume: int,
        target_countries: List[str],
        avg_order_weight_kg: float = 1.5
    ) -> Dict[str, Any]:
        """
        与事件调度模块集成
        为大促活动规划备货和物流
        """
        event_dt = datetime.fromisoformat(event_date)
        today = datetime.now()
        days_until_event = (event_dt - today).days

        planning_report = {
            "event_name": event_name,
            "event_date": event_date,
            "days_until_event": days_until_event,
            "expected_orders": expected_order_volume,
            "total_estimated_weight_kg": expected_order_volume * avg_order_weight_kg,
            "preparation_plan": [],
            "warnings": []
        }

        # 为每个目标国家制定计划
        for country in target_countries:
            logistics = self.calculate_optimal_logistics(
                origin_country="CN",
                destination_country=country,
                weight_kg=planning_report["total_estimated_weight_kg"],
                volume_cbm=planning_report["total_estimated_weight_kg"] * 0.005,
                product_value_usd=expected_order_volume * 30,  # 假设客单价$30
                urgency="urgent" if days_until_event < 15 else "normal"
            )

            if logistics["success"]:
                rec = logistics["recommendation"]

                # 判断是否需要立即行动
                if rec["transit_days"] > days_until_event:
                    planning_report["warnings"].append(
                        f"⚠️ {country}: Transit time ({rec['transit_days']} days) exceeds time until event ({days_until_event} days). "
                        f"Consider overseas warehouse or expedited shipping."
                    )

                    # 检查是否有海外仓选项
                    has_warehouse = any(
                        country in wh["coverage"]
                        for wh in self.overseas_warehouses.values()
                    )

                    if has_warehouse:
                        planning_report["preparation_plan"].append({
                            "country": country,
                            "action": "Use overseas warehouse for last-mile delivery",
                            "deadline": (today + timedelta(days=days_until_event - 3)).strftime("%Y-%m-%d"),
                            "reason": "Standard shipping too slow for event timeline"
                        })
                    else:
                        planning_report["preparation_plan"].append({
                            "country": country,
                            "action": "Book air freight immediately",
                            "deadline": (today + timedelta(days=2)).strftime("%Y-%m-%d"),
                            "reason": "Critical timeline - only fastest option viable"
                        })
                else:
                    planning_report["preparation_plan"].append({
                        "country": country,
                        "action": f"Schedule {rec['mode']} shipment with {rec['provider']}",
                        "deadline": (today + timedelta(days=days_until_event - rec["transit_days"] - 2)).strftime("%Y-%m-%d"),
                        "estimated_cost_usd": rec["shipping_cost_usd"]
                    })

        return planning_report

    def execute_workflow(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        CrawAI框架标准执行接口

        Args:
            task: 任务类型
                - "optimize_logistics": 优化物流方案
                - "calculate_landed_cost": 计算落地成本
                - "predict_inventory": 预测库存
                - "monitor_tariffs": 监控关税
                - "market_feasibility": 市场可行性分析
                - "event_planning": 活动物流规划
            context: 任务上下文参数

        Returns:
            执行结果
        """
        logger.info(f"SupplyChainLogisticsAgent executing task: {task}")

        try:
            if task == "optimize_logistics":
                return self.calculate_optimal_logistics(**context)

            elif task == "calculate_landed_cost":
                landed_cost = self.calculate_landed_cost(**context)
                return {
                    "success": True,
                    "landed_cost_breakdown": landed_cost.to_dict(),
                    "timestamp": datetime.now().isoformat()
                }

            elif task == "predict_inventory":
                prediction = self.predict_inventory_and_replenish(**context)
                return {
                    "success": True,
                    "prediction": asdict(prediction),
                    "alert": prediction.suggested_order_quantity > 0,
                    "timestamp": datetime.now().isoformat()
                }

            elif task == "monitor_tariffs":
                return self.monitor_tariff_changes(**context)

            elif task == "market_feasibility":
                return self.integrate_with_business_development(**context)

            elif task == "event_planning":
                return self.integrate_with_event_scheduler(**context)

            elif task == "sales_integration":
                return self.integrate_with_sales_performance(**context)

            else:
                return {
                    "success": False,
                    "error": f"Unknown task type: {task}",
                    "available_tasks": [
                        "optimize_logistics",
                        "calculate_landed_cost",
                        "predict_inventory",
                        "monitor_tariffs",
                        "market_feasibility",
                        "event_planning",
                        "sales_integration"
                    ]
                }

        except Exception as e:
            logger.error(f"Error executing task {task}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task": task
            }

    def get_agent_info(self) -> Dict[str, Any]:
        """获取Agent信息"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": [
                "Optimal logistics routing (Air/Sea/Rail/Express/Overseas Warehouse)",
                "Landed cost calculation with duty and VAT",
                "Inventory turnover prediction and replenishment alerts",
                "Tariff policy monitoring",
                "Market entry feasibility analysis",
                "Event-driven logistics planning",
                "Integration with Sales Performance, Business Development, and Event Scheduler workflows"
            ],
            "supported_regions": ["US", "EU", "UK", "JP", "AU", "SEA", "CA"],
            "supported_transport_modes": [mode.value for mode in TransportMode],
            "data_sources": [
                "Internal logistics provider database",
                "Tariff rate library (requires periodic updates)",
                "VAT/GST rate database",
                "Overseas warehouse network"
            ],
            "integration_points": [
                "Sales Performance Workflow",
                "Business Development Workflow",
                "Event Scheduler Workflow"
            ]
        }


# 示例使用
if __name__ == "__main__":
    agent = SupplyChainLogisticsAgent()

    # 示例1: 计算最优物流方案
    print("\n=== 示例1: 最优物流方案 ===")
    result = agent.calculate_optimal_logistics(
        origin_country="CN",
        destination_country="US",
        weight_kg=50,
        volume_cbm=0.25,
        product_value_usd=500,
        urgency="normal",
        hs_code="6109"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 示例2: 计算落地成本
    print("\n=== 示例2: 落地成本计算 ===")
    result = agent.execute_workflow(
        task="calculate_landed_cost",
        context={
            "destination_country": "DE",
            "product_value": 200,
            "weight_kg": 5,
            "shipping_cost": 45,
            "hs_code": "6203"
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 示例3: 库存预测
    print("\n=== 示例3: 库存预测与补货 ===")
    sales_history = [
        {"date": f"2024-01-{i:02d}", "quantity_sold": 10 + i % 5}
        for i in range(1, 31)
    ]
    result = agent.execute_workflow(
        task="predict_inventory",
        context={
            "product_sku": "TSHIRT-BLK-M",
            "current_stock": 150,
            "sales_history": sales_history,
            "lead_time_days": 15,
            "safety_stock_days": 15
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 示例4: 大促活动规划
    print("\n=== 示例4: 黑五大促物流规划 ===")
    result = agent.execute_workflow(
        task="event_planning",
        context={
            "event_name": "Black Friday 2024",
            "event_date": "2024-11-29",
            "expected_order_volume": 5000,
            "target_countries": ["US", "UK", "DE", "FR"],
            "avg_order_weight_kg": 1.2
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 示例5: 新市场可行性分析
    print("\n=== 示例5: 新市场进入可行性 ===")
    result = agent.execute_workflow(
        task="market_feasibility",
        context={
            "target_country": "BR",
            "product_categories": ["electronics", "apparel"]
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 打印Agent信息
    print("\n=== Agent信息 ===")
    print(json.dumps(agent.get_agent_info(), indent=2, ensure_ascii=False))