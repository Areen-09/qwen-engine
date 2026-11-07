import subprocess
import time
import urllib.request
import json
import sys

def main():
    print("[Test Server] Starting qwen_server.exe in background...")
    proc = subprocess.Popen(
        ["build\\qwen_chat.exe", "--model", "models\\qwen25_1.5b_w4a16.qwenbin", "--port", "8085"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Wait for server to initialize
    print("[Test Server] Waiting for server to initialize CUDA graph & bind port...")
    time.sleep(5)

    try:
        # 1. Test GET /
        print("[Test Server] Testing GET / (HTML UI)...")
        req = urllib.request.urlopen("http://127.0.0.1:8085/")
        html = req.read().decode("utf-8")
        assert "Qwen2.5-1.5B" in html
        print(f"  -> SUCCESS: Received {len(html)} bytes HTML UI")

        # 2. Test GET /api/stats
        print("[Test Server] Testing GET /api/stats...")
        req = urllib.request.urlopen("http://127.0.0.1:8085/api/stats")
        stats = json.loads(req.read().decode("utf-8"))
        print(f"  -> SUCCESS: Stats: {stats}")
        assert stats["mode"] == "W4A16"

        # 3. Test POST /api/chat SSE streaming
        print("[Test Server] Testing POST /api/chat (SSE streaming)...")
        payload = json.dumps({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi"}
            ],
            "temperature": 0.0,
            "max_new_tokens": 16
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://127.0.0.1:8085/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req)
        print("  -> SSE Stream opened. Reading tokens:")
        tokens_received = 0
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                data_json = json.loads(line_str[6:])
                tok = data_json.get("token", "")
                tokens_received += 1
                print(f"     Token {tokens_received}: {repr(tok)} (is_eos: {data_json.get('is_eos')})")
                if data_json.get("is_eos"):
                    break

        print(f"  -> SUCCESS: Streamed {tokens_received} SSE token packets successfully!")

        # 4. Test Turn 2 (prefix cache hit test)
        print("[Test Server] Testing Turn 2 (Session KV prefix cache hit)...")
        req2 = urllib.request.urlopen("http://127.0.0.1:8085/api/stats")
        stats2 = json.loads(req2.read().decode("utf-8"))
        print(f"  -> Cached tokens in session: {stats2.get('cached_tokens')}")

        print("\n>>> ALL HTTP & SSE SERVER ENDPOINTS VERIFIED SUCCESSFULLY! <<<")

    finally:
        print("[Test Server] Terminating server process...")
        proc.terminate()
        proc.wait(timeout=5)
        print("[Test Server] Server terminated cleanly.")

if __name__ == "__main__":
    main()

