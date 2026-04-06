from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Rule(ABC):
    """
    Base class for all test result validation rules.
    Each rule checks one specific aspect of the FFMPEG conversion result.
    """

    def __init__(self, case, json_content,
                 default_message="No default message specified",
                 description="No description specified"):
        self.case = case
        self.json_content = json_content
        self.default_message = default_message
        self.description = description

    @abstractmethod
    def should_be_executed(self):
        """Return True if this rule applies to the current test case."""
        pass

    @abstractmethod
    def apply(self, data):
        """
        Apply the rule against collected data.
        data: dict with keys: metadata, psnr, ssim, ffmpeg_returncode
        """
        pass

    def add_error(self, message=None):
        """Mark test as failed and record an error message.
        Uses jobs_launcher 'message' list (list of dicts with 'issue'/'description').
        """
        if message is None:
            message = self.default_message
        logger.error(f"Rule failed: {message}")
        self.json_content["message"].append({
            "issue": message,
            "description": self.description
        })
        if self.json_content["test_status"] not in ("error",):
            self.json_content["test_status"] = "failed"

    def add_warning(self, message):
        """Record a warning without changing test status."""
        logger.warning(f"Rule warning: {message}")
        self.json_content["message"].append({
            "issue": f"[Warning] {message}",
            "description": self.description
        })
