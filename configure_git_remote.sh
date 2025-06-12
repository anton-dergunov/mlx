#!/bin/bash

# Usage: ./configure_git_remote.sh <REMOTE_IP> <PORT>
# Example: ./configure_git_remote.sh 201.238.124.65 11046

set -e

# Check arguments
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <REMOTE_IP> <PORT>"
  exit 1
fi

REMOTE_IP="$1"
REMOTE_PORT="$2"
SSH_KEY="$HOME/.ssh/computa"
SSH_USER="root"

# Get local Git config
GIT_USER_NAME=$(git config --global user.name)
GIT_USER_EMAIL=$(git config --global user.email)

if [[ -z "$GIT_USER_NAME" || -z "$GIT_USER_EMAIL" ]]; then
  echo "❌ Git user.name or user.email not configured locally."
  exit 1
fi

echo "🔐 Using SSH key: $SSH_KEY"
echo "🌍 Connecting to $SSH_USER@$REMOTE_IP:$REMOTE_PORT"
echo "🔧 Setting Git config:"
echo "   user.name  = $GIT_USER_NAME"
echo "   user.email = $GIT_USER_EMAIL"

# Run remote commands via SSH
ssh -i "$SSH_KEY" -p "$REMOTE_PORT" "$SSH_USER@$REMOTE_IP" bash <<EOF
  set -e
  git config --global user.name "$GIT_USER_NAME"
  git config --global user.email "$GIT_USER_EMAIL"
  git config --global init.defaultBranch main
  echo "✅ Git config successfully set on remote:"
  git config --global --list
EOF
