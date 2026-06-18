"""Generate the Python gRPC stubs from proto/knowhub.proto.

Run at container build (Dockerfile) and dev start (compose command):
    python scripts/gen_proto.py

Emits knowhub_pb2.py + knowhub_pb2_grpc.py into
src/agent_hub/adapters/grpc/, rewriting the grpc stub's import to be
package-relative so it imports cleanly under agent_hub.adapters.grpc.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "proto"
OUT_DIR = ROOT / "src" / "agent_hub" / "adapters" / "grpc"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        str(PROTO_DIR / "knowhub.proto"),
    ]
    subprocess.run(cmd, check=True)

    # protoc emits `import knowhub_pb2` (top-level); make it package-relative.
    grpc_stub = OUT_DIR / "knowhub_pb2_grpc.py"
    text = grpc_stub.read_text(encoding="utf-8")
    text = re.sub(r"^import knowhub_pb2", "from . import knowhub_pb2", text, flags=re.M)
    grpc_stub.write_text(text, encoding="utf-8")
    print(f"generated stubs in {OUT_DIR}")


if __name__ == "__main__":
    main()
