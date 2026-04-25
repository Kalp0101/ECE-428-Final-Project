# Verify your user ID — the 1000 in the environment paths must match your actual user ID:
id -u pi   # confirm this returns 1000; if not, update both paths above

# Enable and Test
# Reload systemd to pick up the new file
sudo systemctl daemon-reload

# Enable: start automatically on every future boot
sudo systemctl enable argus.service

# Start right now to test (without rebooting)
sudo systemctl start argus.service

# Check status
sudo systemctl status argus.service

# Watch live log output
journalctl -u argus.service -f

# Useful Service Commands
sudo systemctl stop argus.service      # Stop the running service
sudo systemctl restart argus.service   # Restart after making code changes
sudo systemctl disable argus.service   # Prevent autostart (e.g., during development)
journalctl -u argus.service --since "10 minutes ago"   # Recent logs
