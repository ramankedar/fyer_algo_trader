#!/bin/bash
# setup_oracle.sh — One-time setup for ProductionTheta on Oracle Linux / Ubuntu
#
# Run as: bash setup_oracle.sh
# Tested on: Ubuntu 22.04 LTS / Oracle Linux 8 (AMD64 or ARM64)
#
# This script:
#   1. Installs system dependencies
#   2. Creates a Python venv with all algo_trader requirements
#   3. Copies your local code and .env to the server
#   4. Installs the systemd service
#   5. Sets up log rotation and a cron-based market-hours scheduler

set -e

REPO_DIR="$HOME/algo_trader"
VENV_DIR="$REPO_DIR/.venv"
LOG_DIR="$REPO_DIR/logs"
TRADES_DIR="$REPO_DIR/trades"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ProductionTheta Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -q
    sudo apt-get install -y python3 python3-pip python3-venv tmux cron logrotate
elif command -v yum &>/dev/null; then
    sudo yum install -y python3 python3-pip tmux cronie logrotate
fi

# ── 2. Create directories ─────────────────────────────────────────────────────
echo "[2/6] Creating directories..."
mkdir -p "$LOG_DIR" "$TRADES_DIR" "$REPO_DIR/data/cache" "$REPO_DIR/nse_option_cache"

# ── 3. Python venv ────────────────────────────────────────────────────────────
echo "[3/6] Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements_platform.txt" -q
# Install the algo_platform package in editable mode
"$VENV_DIR/bin/pip" install -e "$REPO_DIR" -q
echo "    Venv: $VENV_DIR"

# ── 4. Verify credentials ─────────────────────────────────────────────────────
echo "[4/6] Checking credentials..."
if [ -f "$REPO_DIR/.env" ]; then
    echo "    .env found"
    source "$REPO_DIR/.env"
    [ -n "$BROKER_APP_ID" ]       && echo "    BROKER_APP_ID     : OK" || echo "    BROKER_APP_ID     : MISSING"
    [ -n "$BROKER_ACCESS_TOKEN" ] && echo "    BROKER_ACCESS_TOKEN: OK (token len=${#BROKER_ACCESS_TOKEN})" || echo "    BROKER_ACCESS_TOKEN: MISSING"
else
    echo "    WARNING: .env not found at $REPO_DIR/.env"
    echo "    Copy your local .env to the server before starting."
fi

# ── 5. Systemd service ────────────────────────────────────────────────────────
echo "[5/6] Installing systemd service..."
sudo cp "$REPO_DIR/theta.service" /etc/systemd/system/theta.service
# Update the user in the service file to match current user
sudo sed -i "s/User=ubuntu/User=$(whoami)/" /etc/systemd/system/theta.service
sudo sed -i "s/Group=ubuntu/Group=$(whoami)/" /etc/systemd/system/theta.service
sudo sed -i "s|/home/ubuntu/algo_trader|$REPO_DIR|g" /etc/systemd/system/theta.service
sudo systemctl daemon-reload
echo "    Service installed: theta.service"
echo "    Start with: sudo systemctl start theta"
echo "    Enable auto-start: sudo systemctl enable theta"

# ── 6. Cron for market hours ──────────────────────────────────────────────────
echo "[6/6] Setting up cron job (Thursday 9:00 AM IST = 03:30 UTC)..."
CRON_LINE="30 3 * * 4 cd $REPO_DIR && $VENV_DIR/bin/python3 run_live_theta.py >> $LOG_DIR/cron.log 2>&1"
# Add to crontab if not already there
( crontab -l 2>/dev/null | grep -v "run_live_theta" ; echo "$CRON_LINE" ) | crontab -
echo "    Cron installed: runs every Thursday at 09:00 IST"

# ── Log rotation ──────────────────────────────────────────────────────────────
sudo tee /etc/logrotate.d/theta > /dev/null <<EOF
$LOG_DIR/*.log {
    weekly
    rotate 12
    compress
    missingok
    notifempty
    copytruncate
}
EOF
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete."
echo ""
echo "  Quick test (dry-run, no orders):"
echo "    $VENV_DIR/bin/python3 $REPO_DIR/run_live_theta.py --dry-run"
echo ""
echo "  Paper trading (Thursday only, no real orders):"
echo "    $VENV_DIR/bin/python3 $REPO_DIR/run_live_theta.py"
echo ""
echo "  Monitor trades (separate terminal):"
echo "    $VENV_DIR/bin/python3 $REPO_DIR/monitor_theta.py --watch"
echo ""
echo "  SSH monitoring shortcut (add to ~/.bashrc):"
echo "    alias theta='python3 $REPO_DIR/monitor_theta.py --all'"
echo "    alias theta-log='tail -f $LOG_DIR/theta_\$(date +%Y-%m-%d).log'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
