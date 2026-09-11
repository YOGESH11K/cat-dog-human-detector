import json
import os
import sys
from http import HTTPStatus

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _predictor import decode_image, load_class_names, predict_image  # noqa: E402


def handler(request):
    try:
        body = request.get_json()
        if not body or not body.get("image"):
            return (
                json.dumps({"detail": "Please provide a base64 image string in the 'image' field."}),
                HTTPStatus.BAD_REQUEST,
                {"Content-Type": "application/json"},
            )
        img = decode_image(body["image"])
        class_names = load_class_names()
        result = predict_image(img, class_names)
        return (
            json.dumps(result),
            HTTPStatus.OK,
            {"Content-Type": "application/json"},
        )
    except Exception as exc:
        return (
            json.dumps({"detail": f"Could not process image: {exc}"}),
            HTTPStatus.BAD_REQUEST,
            {"Content-Type": "application/json"},
        )