"""用 httpx 直连的客户端示例 —— 不依赖 typesafe_sdk，方便看清线上到底发了什么。

    .venv/bin/python examples/client.py
    BASE_URL=http://127.0.0.1:8077 .venv/bin/python examples/client.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8077")
API_KEY = os.environ.get("API_KEY")
DIR = Path(__file__).parent


def call(client: httpx.Client, payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    response = client.post("/v1/systemone", json=payload, headers=headers)
    if response.status_code != 200:
        # 错误体统一是 {"error": {"code", "message"}}，422 还会多一个逐字段的 detail。
        raise SystemExit(f"{response.status_code}: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    return response.json()


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        models = client.get("/v1/models").json()
        print("可用模型：")
        for item in models["objects"]:
            mark = "*" if item["alias"] == models["default"] else " "
            print(f"  {mark} {item['alias']:<26} → {item['slot']:<17} {item['checkpoint']}")

        for name in ("request.json", "request_zh.json"):
            payload = json.loads((DIR / name).read_text(encoding="utf-8"))
            body = call(client, payload)
            print(f"\n=== {name} → {body['model']} ===")
            for qid, answer in body["answers"].items():
                if answer["type"] == "choice":
                    print(f"  {qid:<12} choice  {answer['choice']!r} (confidence {answer['confidence']})")
                elif answer["type"] == "score":
                    print(f"  {qid:<12} score   {answer['score']} / {len(answer['legend']) - 1}")
                else:
                    print(f"  {qid:<12} noul    {answer['noul']}")
            print(f"  usage: {body['usage']}")


if __name__ == "__main__":
    main()
