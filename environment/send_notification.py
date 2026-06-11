"""
MethodManager 'Run Exe' script to send an SMS notification.
Usage: python send_notification.py "Assembly complete"
Exits 0 on success, 1 on failure.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from notify import send_notification


def main():
    if len(sys.argv) < 2:
        print("Usage: send_notification.py <message>")
        sys.exit(1)

    message = " ".join(sys.argv[1:])
    print(f"Sending notification: {message}")

    if send_notification(message):
        print("Notification sent successfully.")
        sys.exit(0)
    else:
        print("Notification send failed (see log for details).")
        sys.exit(1)


if __name__ == "__main__":
    main()
