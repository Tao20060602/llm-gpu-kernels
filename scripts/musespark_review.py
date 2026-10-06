"""User-authorized external method review; credentials stay in environment."""
import json
import os
from pathlib import Path
import urllib.request
import urllib.error
import uuid


def main():
    directory = Path("artifacts/m3/musespark")
    directory.mkdir(parents=True, exist_ok=True)
    session_file = directory / "session_id.txt"
    if not session_file.exists():
        session_file.write_text(str(uuid.uuid4()))
    prompt = """Review this GPU experiment plan, in Chinese. Do not propose unverified speedups.
Hardware: RTX3080 Laptop SM86 16GiB. Python3.12, torch2.7.1 cu128, Triton3.3.1.
Task: forward BF16 ragged grouped GEMM, FP32 accumulator; A[sumM,K], W[E,N,K], C_e=A_e@W_e.T.
Host-known counts and GPU task metadata prepared outside timed kernel. No full MoE/routing cost measured.
Fixed CUTLASS3.8 classic DeviceOnly baseline CTA64x128x32, warp32x64x32, stages3 align8, and original Triton BM32 BN64 BK32 warps4 stages3.
Qwen gate/up shape E128 N1536 K2048. Graph p50 T512 balanced: Triton2.835ms CUTLASS1.963ms; T4096:9.704ms vs6.708ms.
Small E8/N256/K128 Graph: Triton2.8us CUTLASS6.5us, but Python API Triton33.7us CUTLASS7.9us.
Proposed M3: cached prepared launcher validated once, retain buffers and choose current stream on each run; no hidden counts readback/allocation. Triton internal API version pinned.
Tile sweep on training cases (T64,T512,T4096 balanced; T512 hotspot); BM16/32/64, BN64/128/256, BK32/64, warps4/8, stages2/3/4, bounded subset. Validate every candidate against FP32 oracle before timing, preserve failed configs.
Choose configurations only from training; freeze rule before holdout T17/T193/T1024/T2049 with random distinct topk routes, new seeds, and down GEMM shape N2048 K768. Repeat old/tuned/CUTLASS matched graph A/B in balanced cyclic orders. Save all samples, p95, telemetry, hashes.
No ncu hardware counters available; clock/power not fixed. Hot repeated buffers, not serving latency.
Give: (1) three testable hypotheses and evidence needed (2) highest-risk methodological errors (3) reasonable simple selection rule features that avoid per-holdout tuning. Keep <=900 Chinese characters. You have no tools or repository access; do not claim experiments were run."""
    body = {"model": "muse-spark-1.3-contributor", "input": prompt, "max_output_tokens": 6000,
            "reasoning": {"effort": "low"}, "store": False}
    request = urllib.request.Request("https://opencode.ai/zen/go/v1/responses", data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + os.environ["OPENCODE_GO_API_KEY"],
                                              "Content-Type": "application/json", "Accept": "application/json",
                                              "User-Agent": "llm-gpu-kernels/0.1 API client",
                                              "x-opencode-session": session_file.read_text().strip()}, method="POST")
    (directory / "request.json").write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read())
        (directory / "response.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        texts = [item.get("text", "") for output in data.get("output", []) for item in output.get("content", []) if item.get("type") == "output_text"]
        answer = "\n".join(texts)
        (directory / "review.md").write_text(answer + "\n")
        print(json.dumps({"status": data.get("status"), "model": data.get("model"), "usage": data.get("usage"), "answer": answer}, ensure_ascii=False))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace").replace(os.environ["OPENCODE_GO_API_KEY"], "[redacted]")
        try:
            error = json.loads(raw).get("error", {})
            detail = {key: error.get(key) for key in ("code", "type", "message") if isinstance(error, dict) and key in error}
        except ValueError:
            detail = {"non_json_body": True}
        failure = {"status": "provider_http_error", "http_status": exc.code, "error": detail}
        print(json.dumps(failure, ensure_ascii=False))
        (directory / "failure.json").write_text(json.dumps(failure, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
