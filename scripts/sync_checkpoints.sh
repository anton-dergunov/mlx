#!/bin/bash

# MacOS comes with old vesion of rsync. Install more recent rsync locally and verify:
# % brew install rsync
# % which rsync
# /opt/homebrew/bin/rsync
# % rsync --version
# should be > 3.x
#
# Also install rsync on Linux (setup_runpod.sh does it)

set -e

usage() {
  echo "Usage: $0 -h <remote_host> -p <ssh_port> -r <remote_dir> -l <local_dir>"
  echo
  echo "Example:"
  echo "  $0 -h 142.169.249.42 -p 21506 -r /root/checkpoints -l /Users/you/CheckpointsBackup"
  exit 1
}

# Fixed SSH config
REMOTE_USER=root
SSH_KEY=~/.ssh/computa
SSH_PORT=22

# Parse flags
while getopts ":h:p:r:l:" opt; do
  case ${opt} in
    h )
      REMOTE_HOST=$OPTARG
      ;;
    p )
      SSH_PORT=$OPTARG
      ;;
    r )
      REMOTE_DIR=$OPTARG
      ;;
    l )
      LOCAL_DIR=$OPTARG
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    : )
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Validate
if [[ -z "$REMOTE_HOST" || -z "$REMOTE_DIR" || -z "$LOCAL_DIR" ]]; then
  echo "Missing required argument(s)." >&2
  usage
fi

echo "-------------------------------------------"
echo " Remote host: $REMOTE_HOST"
echo " SSH port:    $SSH_PORT"
echo " Remote dir:  $REMOTE_DIR"
echo " Local dir:   $LOCAL_DIR"
echo " SSH key:     $SSH_KEY"
echo "-------------------------------------------"

MIN_AGE_MINUTES=1  # Minimum file age to avoid partials

while true; do
  echo "[`date`] Finding files older than $MIN_AGE_MINUTES minute(s)..."

  ssh -p "$SSH_PORT" -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
    "cd $REMOTE_DIR && find . -type f -mmin +$MIN_AGE_MINUTES" > files_to_sync.txt

  echo "[`date`] Syncing files..."
  rsync -avz --files-from=files_to_sync.txt --relative --progress \
    --partial --append-verify \
    -e "ssh -p $SSH_PORT -i $SSH_KEY" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR"

  echo "[`date`] Waiting 5 minutes before next sync..."
  sleep 300
done
