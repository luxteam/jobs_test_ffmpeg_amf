from rules.rule import Rule
import logging

logger = logging.getLogger(__name__)


class MetadataRule(Rule):
    """
    Checks that converted video metadata matches expected values from the test case.
    Reports each mismatched field individually.
    """

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="Converted video metadata does not match expected values",
            description="Validate ffprobe metadata of output video against test case expected_metadata"
        )

    def should_be_executed(self):
        return "expected_metadata" in self.case and bool(self.case["expected_metadata"])

    def apply(self, data):
        actual = data.get("metadata", {})
        expected = self.case.get("expected_metadata", {})

        all_ok = True
        for field, expected_value in expected.items():
            actual_value = actual.get(field)
            if actual_value is None:
                self.add_error(
                    f"Metadata field '{field}' not found in output video. Expected: {expected_value}"
                )
                all_ok = False
            elif str(actual_value) != str(expected_value):
                self.add_error(
                    f"Metadata mismatch for '{field}': expected={expected_value}, actual={actual_value}"
                )
                all_ok = False

        if all_ok:
            logger.info("MetadataRule: all metadata fields match")


class PSNRRule(Rule):
    """
    Checks PSNR (Peak Signal-to-Noise Ratio) between input and output video.
    PSNR below threshold indicates unacceptable quality degradation.
    Default threshold: 30.0 dB (overridable via case "psnr_threshold" field).
    Skipped for cases without a reference input_video (e.g. lavfi-generated source).
    """

    DEFAULT_THRESHOLD = 30.0

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="PSNR value is below acceptable threshold",
            description="Compare PSNR between input and output video using FFMPEG psnr filter"
        )

    def should_be_executed(self):
        return bool(self.case.get("input_video"))

    def apply(self, data):
        psnr = data.get("psnr")
        threshold = self.case.get("psnr_threshold", self.DEFAULT_THRESHOLD)

        if psnr is None:
            self.add_error("PSNR measurement failed — no value returned by ffmpeg")
            return

        if psnr == float("inf"):
            logger.info(f"PSNRRule: PSNR is infinite (lossless or identical), threshold={threshold}")
            return

        if psnr < threshold:
            self.add_error(
                f"Unacceptable PSNR: {psnr:.2f} dB (threshold: {threshold} dB)"
            )
        else:
            logger.info(f"PSNRRule: PSNR={psnr:.2f} dB >= threshold={threshold} dB — OK")


class SSIMRule(Rule):
    """
    Checks SSIM (Structural Similarity Index) between input and output video.
    SSIM below threshold indicates unacceptable quality degradation.
    Range: 0.0 to 1.0. Default threshold: 0.9 (overridable via case "ssim_threshold" field).
    Skipped for cases without a reference input_video (e.g. lavfi-generated source).
    """

    DEFAULT_THRESHOLD = 0.9

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="SSIM value is below acceptable threshold",
            description="Compare SSIM between input and output video using FFMPEG ssim filter"
        )

    def should_be_executed(self):
        return bool(self.case.get("input_video"))

    def apply(self, data):
        ssim = data.get("ssim")
        threshold = self.case.get("ssim_threshold", self.DEFAULT_THRESHOLD)

        if ssim is None:
            self.add_error("SSIM measurement failed — no value returned by ffmpeg")
            return

        if ssim < threshold:
            self.add_error(
                f"Unacceptable SSIM: {ssim:.4f} (threshold: {threshold})"
            )
        else:
            logger.info(f"SSIMRule: SSIM={ssim:.4f} >= threshold={threshold} — OK")


class ConversionSuccessRule(Rule):
    """
    Checks that the FFMPEG conversion process completed without error.
    Always applied to every test case.
    """

    def __init__(self, case, json_content):
        super().__init__(
            case, json_content,
            default_message="FFMPEG conversion process returned a non-zero exit code",
            description="Check that ffmpeg exited successfully (return code 0)"
        )

    def should_be_executed(self):
        return True

    def apply(self, data):
        returncode = data.get("ffmpeg_returncode")
        if returncode is None or returncode != 0:
            self.add_error(
                f"FFMPEG conversion failed with exit code: {returncode}"
            )
        else:
            logger.info("ConversionSuccessRule: ffmpeg exited successfully")
