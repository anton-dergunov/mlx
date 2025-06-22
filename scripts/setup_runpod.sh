#!/usr/bin/env bash
set -euo pipefail

SSH_CONFIG="$HOME/.ssh/config"
HOST_ALIAS="RemoteRunPod"

usage() {
  cat <<EOF
Usage: $0 [-h host] [-p port]
  -h host    IP or hostname for RemoteRunPod
  -p port    SSH port for RemoteRunPod
If either -h or -p is given, ~/.ssh/config Host $HOST_ALIAS will be updated.
EOF
  exit 1
}

# ─── Parse args ───────────────────────────────────────────────────────────────
OVERRIDE_HOST=""
OVERRIDE_PORT=""
while getopts "h:p:" opt; do
  case $opt in
    h) OVERRIDE_HOST="$OPTARG" ;;
    p) OVERRIDE_PORT="$OPTARG" ;;
    *) usage ;;
  esac
done

echo
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                    RunPod Setup Script v1.0                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo

# ─── Optionally patch ~/.ssh/config ─────────────────────────────────────────
if [[ -n "$OVERRIDE_HOST" || -n "$OVERRIDE_PORT" ]]; then
  echo "🔧 Updating SSH config for Host $HOST_ALIAS..."
  # backup
  cp "$SSH_CONFIG" "${SSH_CONFIG}.bak.$(date +%s)" 
  # use awk to rewrite or insert Host entry
  awk -v host="$OVERRIDE_HOST" -v port="$OVERRIDE_PORT" '
    /^Host '"$HOST_ALIAS"'$/{ in_block=1; print; next }
    in_block && /^[[:space:]]*HostName / && host!="" {
      sub($2,host); in_block=2
    }
    in_block && /^[[:space:]]*Port / && port!="" {
      sub($2,port); in_block=2
    }
    in_block && /^[[:space:]]*IdentityFile / {
      # leave IdentityFile untouched
    }
    in_block && /^[[:space:]]*Host / && in_block==2 { in_block=0 }
    { print }
  ' "$SSH_CONFIG" > "${SSH_CONFIG}.new"
  mv "${SSH_CONFIG}.new" "$SSH_CONFIG"
  echo "✅ SSH config updated (backup at ${SSH_CONFIG}.bak.*)"
fi

# ─── Extract SSH parameters from config ───────────────────────────────────────
echo
echo "📖 Reading SSH parameters from $SSH_CONFIG for Host $HOST_ALIAS..."
REMOTE_HOST=$(awk "/^Host $HOST_ALIAS\$/{f=1} f && /^[[:space:]]*HostName /{print \$2; exit}" "$SSH_CONFIG")
REMOTE_PORT=$(awk "/^Host $HOST_ALIAS\$/{f=1} f && /^[[:space:]]*Port /{print \$2; exit}" "$SSH_CONFIG")
SSH_KEY=$(awk "/^Host $HOST_ALIAS\$/{f=1} f && /^[[:space:]]*IdentityFile /{print \$2; exit}" "$SSH_CONFIG")

if [[ -z "$REMOTE_HOST" || -z "$REMOTE_PORT" || -z "$SSH_KEY" ]]; then
  echo "❌ Missing HostName/Port/IdentityFile for $HOST_ALIAS in $SSH_CONFIG"
  exit 1
fi

echo "🌐  Host: $REMOTE_HOST"
echo "🔌  Port: $REMOTE_PORT"
echo "🔑  Key : $SSH_KEY"

# ─── Git user info for remote ────────────────────────────────────────────────
echo
echo "📋 Checking local Git config..."
GIT_USER_NAME=$(git config --global user.name || true)
GIT_USER_EMAIL=$(git config --global user.email || true)

if [[ -z "$GIT_USER_NAME" || -z "$GIT_USER_EMAIL" ]]; then
  echo "❌ Please set git user.name and user.email locally first."
  exit 1
fi

echo "🧑‍💻  Will configure on remote:"
echo "      user.name  = $GIT_USER_NAME"
echo "      user.email = $GIT_USER_EMAIL"

# ─── SSH & remote setup ─────────────────────────────────────────────────────
echo
echo "🔗 Connecting to remote and setting up environment..."

ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$REMOTE_PORT" "root@$REMOTE_HOST" bash <<EOF
GIT_USER_NAME="$GIT_USER_NAME"
GIT_USER_EMAIL="$GIT_USER_EMAIL"

set -euo pipefail

print_step() {
  local step_number="\$1"
  local step_label="\$2"
  local prefix="🚀 STEP \$step_number: "
  local full_text="\${prefix}\${step_label}"

  echo -e "\\n\$full_text"
  local len=\$(echo -n "\$full_text" | wc -m | awk '{print \$1}')
  printf '%*s\\n' "\$len" '' | tr ' ' '-'
}

print_step 1 "Configuring Git on remote..."
git config --global user.name "\$GIT_USER_NAME"
git config --global user.email "\$GIT_USER_EMAIL"
git config --global init.defaultBranch main
git config --global pull.rebase false

print_step 2 "Cloning/updating project repository..."
cd /root
if [[ ! -d "mlx" ]]; then
  git clone https://github.com/anton-dergunov/mlx.git
else
  cd mlx && git pull
fi

print_step 3 "Creating & activating Python venv..."
# adjust path as needed
# python3 -m venv ~/venv
# source ~/venv/bin/activate

print_step 4 "Installing Python dependencies"
# pip install --upgrade pip
# pip install ipykernel notebook jupyterlab ipywidgets
pip install --upgrade pip
pip install --ignore-installed ipykernel notebook jupyterlab ipywidgets

echo
echo "✅ Remote environment setup complete!"
EOF

echo
echo "🎉 All done! You can now SSH: ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST"

echo
echo "🧑🏻‍💻 Starting VS Code..."

code --remote ssh-remote+RemoteRunPod /root/mlx
