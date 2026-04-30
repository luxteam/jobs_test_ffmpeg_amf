import logging
import traceback

from rules.rule_impl.ffmpeg_rules import (
    ConversionSuccessRule,
    DecodeRule,
    FormatRule,
    FrameCountRule,
    MetadataRule,
    PSNRRule,
    SSIMRule,
)

logger = logging.getLogger(__name__)

# Registry: rule name in test case JSON -> Rule class
RULES = {
    "conversion_success": ConversionSuccessRule,
    "decode_rule":        DecodeRule,
    "format_rule":        FormatRule,
    "frame_count_rule":   FrameCountRule,
    "metadata_rule":      MetadataRule,
    "psnr_rule":          PSNRRule,
    "ssim_rule":          SSIMRule,
}

# Rules applied to every test case regardless of case["rules"]
GENERAL_RULES = ["conversion_success"]


class RulesProcessor:
    def __init__(self, case, json_content):
        self.case = case
        self.json_content = json_content
        self.rules = []

        # Always attach general rules
        for rule_name in GENERAL_RULES:
            self.rules.append(RULES[rule_name](case, json_content))

        # Attach case-specific rules listed in test case JSON
        for rule_name in case.get("rules", []):
            if rule_name in RULES:
                self.rules.append(RULES[rule_name](case, json_content))
            else:
                logger.warning(f"Unknown rule '{rule_name}' in case '{case['case']}' - skipping")

    def process(self, context):
        """
        Apply all applicable rules. Each rule collects its own data from context.
        context: dict with paths/runtime values passed from the test runner.
        """
        try:
            for rule in self.rules:
                if rule.should_be_executed():
                    logger.info(f"Applying rule: {rule.__class__.__name__}")
                    rule.apply(context)
        except Exception as e:
            logger.error(f"Unexpected error in rules_processor: {e}")
            logger.error(traceback.format_exc())
            self.json_content["test_status"] = "error"
            self.json_content["message"].append({
                "issue": f"Rules processor crashed: {e}",
                "description": "Internal error during rule evaluation"
            })
