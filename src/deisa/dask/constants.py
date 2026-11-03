from typing import Final

FEEDBACK_QUEUE_PREFIX: Final[str] = "deisa_feedback_queue_"
# The two queues of the on-demand reduction path: ``_PLAN`` carries the analytics host's requests to the bridges,
# ``_REPORT`` carries the per-bridge partials back. Both are named per array, like the feedback queues.
ONDEMAND_PLAN_QUEUE_PREFIX: Final[str] = "deisa_ondemand_plan_"
ONDEMAND_REPORT_QUEUE_PREFIX: Final[str] = "deisa_ondemand_report_"
CALLBACK_PREFIX: Final[str] = "deisa_cb_"
DEFAULT_SLIDING_WINDOW_SIZE: int = 1
CLIENT_KEY: Final[str] = "deisa"
KEY_PREFIX: Final[str] = "deisa"
WAIT_FOR_EXECUTE_CB_EVENT: Final[str] = "deisa_exec_cb_go"
