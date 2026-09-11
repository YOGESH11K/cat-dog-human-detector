import json
import os
import sys
from http import HTTPStatus

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _predictor import load_class_names  # noqa: E402


def handler(request):
    return (
        json.dumps(
            {
                "name": "Cat / Dog / Human Detector API",
                "classes": load_class_names(),
                "endpoints": ["/api/predict", "/api/health"],
            }
        ),
        HTTPStatus.OK,
        {"Content-Type": "application/json"},
    )