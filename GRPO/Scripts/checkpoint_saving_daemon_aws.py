import os
import re
import sys
import time
import signal
import urllib.request
from huggingface_hub import HfApi

CHECK_INTERVAL = 5
CHECKPOINT_ROOT = "compiler-llvm-deepseek-8b-grpo-checkpoints"  # must match output_dir in run_phase4_v5.py
HF_REPO = "bcoding/compiler-llvm-deepseek-8b-grpo-checkpoints"
POST_SIGINT_WAIT = 10  # seconds to let the trainer flush before we look for a checkpoint

def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

def get_token():
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return None


def is_interrupted(token):
    try:
        headers = {"X-aws-ec2-metadata-token": token} if token else {}
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/spot/instance-action",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_latest_checkpoint(root):
    """Look inside `root` for subfolders named checkpoint-<N> (how HF
    Trainer/TRL names them) and return the one with the largest N.
    Falls back to `root` itself if no numbered subfolders are found."""
    if not os.path.isdir(root):
        return None
    best_step, best_path = -1, None
    for name in os.listdir(root):
        m = re.match(r"checkpoint-(\d+)$", name)
        full = os.path.join(root, name)
        if m and os.path.isdir(full):
            step = int(m.group(1))
            if step > best_step:
                best_step, best_path = step, full
    return best_path or root


def upload_to_hf(folder_path, repo_id):
    print(f"\n[SpotDaemon] Uploading {folder_path} to Hugging Face ({repo_id})...")
    try:
        api = HfApi()
        
        # Check if the repo exists before trying to create it
        if not api.repo_exists(repo_id=repo_id, repo_type="model"):
            print(f"[SpotDaemon] Repository {repo_id} not found. Creating it now...")
            api.create_repo(repo_id=repo_id, repo_type="model", private=True)
        
        # Upload the folder contents
        api.upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type="model")
        print("✅ [SpotDaemon] Upload complete!")
    except Exception as e:
        print(f"❌ [SpotDaemon] Upload failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python spot_daemon.py <PID_OF_TRAINING_SCRIPT>")
        sys.exit(1)

    train_pid = int(sys.argv[1])
    print(f"[SpotDaemon] Monitoring AWS Spot status for Training PID {train_pid}...")
    token = get_token()

    while True:
        # 1. First, check if AWS is pulling the plug
        if is_interrupted(token):
            print("\n🚨 [SpotDaemon] SPOT INTERRUPTION NOTICE DETECTED! 🚨")
            try:
                os.kill(train_pid, signal.SIGINT)
                print(f"[SpotDaemon] Sent SIGINT to {train_pid}, waiting {POST_SIGINT_WAIT}s for checkpoint flush...")
            except ProcessLookupError:
                print(f"[SpotDaemon] PID {train_pid} already gone.")

            time.sleep(POST_SIGINT_WAIT)

            ckpt = find_latest_checkpoint(CHECKPOINT_ROOT)
            if ckpt:
                upload_to_hf(ckpt, HF_REPO)
            else:
                print(f"[SpotDaemon] No checkpoint folder found under {CHECKPOINT_ROOT}")
            sys.exit(0)

        # 2. Next, check if the training script died on its own (OOM, crash, finished)
        if not process_alive(train_pid):
            print(f"❌ [SpotDaemon] Training process {train_pid} is no longer running. Exiting to trigger systemd restart.")
            sys.exit(1)  # Non-zero exit tells systemd this was a failure

        # 3. Wait and check again
        time.sleep(CHECK_INTERVAL)