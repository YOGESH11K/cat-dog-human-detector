import json
import os
import sys
from http import HTTPStatus

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _predictor import MODEL_PATH, load_class_names  # noqa: E402


def handler(request):
    try:
        load_class_names()
        model_loaded = os.path.exists(MODEL_PATH)
        return (
            json.dumps(
                {
                    "status": "ok" if model_loaded else "degraded",
                    "model_loaded": model_loaded,
                    "model": os.path.basename(MODEL_PATH),
                    "classes": load_class_names(),
                }
            ),
            HTTPStatus.OK,
            {"Content-Type": "application/json"},
        )
    except Exception as exc:
        return (
            json.dumps({"status": "error", "detail": str(exc)}),
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {"Content-Type": "application/json"},
        )