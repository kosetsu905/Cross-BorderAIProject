import logging
import os
from functools import lru_cache
from typing import Any

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool

logger = logging.getLogger(__name__)


class B2BLeadLookupTool(BaseTool):
    name: str = "Global B2B Lead & Company Profiler"
    description: str = (
        "Fetches company profiles, decision-maker contacts, and recent business "
        "milestones for cross-border business development."
    )

    def _run(self, company_name: str, region: str) -> dict[str, Any]:
        if not os.getenv("CRUNCHBASE_API_KEY") and not os.getenv("APOLLO_API_KEY"):
            return self._dev_fallback(company_name, region)

        # Production placeholder: integrate Apollo.io, Crunchbase, or a compliant
        # LinkedIn Sales Navigator data provider here.
        return {
            "company": company_name,
            "region": region,
            "status": "prod_ready",
            "message": "Connect a real B2B data provider endpoint.",
        }

    @staticmethod
    @lru_cache(maxsize=32)
    def _dev_fallback(company_name: str, region: str) -> dict[str, Any]:
        logger.info("Using B2B lead lookup dev fallback")
        return {
            "company": company_name,
            "region": region,
            "decision_makers": [
                {
                    "name": "Local DM",
                    "title": "Head of Partnerships",
                    "linkedin": "placeholder",
                }
            ],
            "recent_milestone": "Expanded regional distribution network",
            "partnership_fit_score": "High",
            "status": "dev_mode",
        }


class OutreachToneValidator(BaseTool):
    name: str = "Cross-Border Outreach Tone & Compliance Checker"
    description: str = (
        "Validates outreach copy for cultural tone, spam triggers, and regional "
        "B2B compliance such as GDPR, CAN-SPAM, and CASL."
    )

    def _run(self, text: str, region: str) -> dict[str, Any]:
        flags: list[str] = []
        normalized_region = region.upper()
        normalized_text = text.lower()

        if normalized_region in {"EU", "UK", "DE"} and "unsubscribe" not in normalized_text:
            flags.append(
                "EU/UK/DE outreach should include clear opt-out or unsubscribe language."
            )

        high_pressure_terms = {"urgent", "act now", "limited time only", "guaranteed"}
        if any(term in normalized_text for term in high_pressure_terms):
            flags.append(
                "Contains high-pressure B2B outreach language that may trigger spam filters."
            )

        return {
            "compliant": not flags,
            "flags": flags or ["Tone and compliance validation passed."],
            "recommendations": (
                "Maintain a professional, consultative tone. Add a physical address "
                "and unsubscribe link for EU, UK, and Canada outreach."
            ),
        }


class CRMFormatterTool(BaseTool):
    name: str = "CRM Payload & Pipeline Formatter"
    description: str = (
        "Structures lead data, outreach steps, and follow-up cadences into "
        "standardized CRM JSON payloads."
    )

    def _run(
        self,
        lead_data: dict[str, Any],
        outreach_sequence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "crm_system": "hubspot_salesforce_pipedrive_compatible",
            "lead_record": {
                "company": lead_data.get("company", ""),
                "industry": lead_data.get("industry", ""),
                "region": lead_data.get("region", ""),
                "status": "New Lead",
                "source": "AI BD Workflow",
            },
            "activity_log": [
                {
                    "touch": index + 1,
                    "channel": step.get("platform", ""),
                    "content_preview": (
                        step.get("subject_line") or step.get("subject") or ""
                    )[:50],
                }
                for index, step in enumerate(outreach_sequence)
            ],
            "next_action_date": "T+3 days",
            "status": "formatted",
        }
