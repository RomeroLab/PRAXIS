"""
Push notification via Pushover API.
Uses only stdlib (urllib).
"""

import os
import json
import logging
import urllib.request
import urllib.parse
import yaml

logger = logging.getLogger('lab_automation')

_config = None

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"


def _load_config(config_path='configs/lab_config.yml'):
    global _config
    if _config is not None:
        return _config

    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    _config = full_config.get('notifications', {})
    return _config


def send_notification(message, config_path='configs/lab_config.yml'):
    """Send a push notification via Pushover. Returns True on success, False on failure."""
    try:
        cfg = _load_config(config_path)

        if not cfg.get('enabled', False):
            logger.debug("Notifications disabled in config")
            return False

        data = urllib.parse.urlencode({
            'token': cfg['pushover_api_token'],
            'user': cfg['pushover_user_key'],
            'message': message[:1024],
        }).encode()

        req = urllib.request.Request(PUSHOVER_API_URL, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())

        if result.get('status') == 1:
            logger.info(f"Notification sent: {message}")
            return True
        else:
            logger.error(f"Pushover returned errors: {result.get('errors')}")
            return False

    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        return False


# Backward-compatible alias
send_sms = send_notification
