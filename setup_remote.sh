#!/bin/bash

set -e

# === Resolve SSH config values for Host "RemoteRunPod" ===
SSH_CONFIG=~/.ssh/config
HOST_ALIAS="RemoteRunPod"

REMOTE_HOST=$(awk "/^Host $HOST_ALIAS\$/{f=1} f && /^ *HostName /{print \$2; exit}" "$SSH_CONFIG")
REMOTE_PORT=$(awk "/^Host $HOST_ALIAS\$/{f=1} f && /^ *Port /{print \$2; exit}" "$SSH_CONFIG")
SSH_KEY=$(awk "/^Host $HOST_ALIAS\$/{f=1} f && /^ *IdentityFile /{print \$2; exit}" "$SSH_CONFIG")

if [[ -z "$REMOTE_HOST" || -z "$REMOTE_PORT" || -z "$SSH_KEY" ]]; then
  echo "❌ Could not find all required fields (HostName, Port, IdentityFile) in $SSH_CONFIG under Host $HOST_ALIAS"
  exit 1
fi

echo "🌐 Remote IP: $REMOTE_HOST"
echo "🔐 SSH Key: $SSH_KEY"
echo "🔌 Port: $REMOTE_PORT"

# === Get local Git config ===
GIT_USER_NAME=$(git config --global user.name)
GIT_USER_EMAIL=$(git config --global user.email)

if [[ -z "$GIT_USER_NAME" || -z "$GIT_USER_EMAIL" ]]; then
  echo "❌ Missing local Git config: user.name or user.email"
  exit 1
fi

echo "🧾 Git config to set remotely:"
echo "   user.name  = $GIT_USER_NAME"
echo "   user.email = $GIT_USER_EMAIL"

# === SSH and set Git config + clone repo ===
ssh -i "$SSH_KEY" -p "$REMOTE_PORT" "root@$REMOTE_HOST" bash <<EOF
  set -e
  git config --global user.name "$GIT_USER_NAME"
  git config --global user.email "$GIT_USER_EMAIL"
  git config --global init.defaultBranch main

  if [ ! -d "mlx" ]; then
    git clone https://github.com/anton-dergunov/mlx.git
  else
    echo "📁 mlx repo already exists; skipping clone"
  fi

  mkdir -p /root/mlx/data
  echo "✅ Git configured and repo cloned."
EOF

# === Copy data directory contents to remote ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DATA_DIR="$SCRIPT_DIR/data"

if [ ! -d "$LOCAL_DATA_DIR" ]; then
  echo "❌ Local data directory not found at $LOCAL_DATA_DIR"
  exit 1
fi

echo "📤 Copying files from $LOCAL_DATA_DIR to remote:/root/mlx/data ..."
scp -i "$SSH_KEY" -P "$REMOTE_PORT" "$LOCAL_DATA_DIR"/* "root@$REMOTE_HOST:/root/mlx/data/"

echo "✅ All done!"
