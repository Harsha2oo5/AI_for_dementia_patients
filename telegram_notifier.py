"""
telegram_notifier.py
====================
Place this file in your project root:  FACE/telegram_notifier.py

HOW TO SET UP TELEGRAM 2-STEP VERIFICATION
==========================================

STEP 1 — Create your bot (2 min)
─────────────────────────────────
1. Open Telegram on your phone
2. Search for @BotFather
3. Send:  /newbot
4. Give it a name:     MindBridge Verifier
5. Give it a username: mindbridge_verify_bot   (must end in 'bot')
6. BotFather replies with your TOKEN — copy it, looks like:
   7123456789:AAGx8_xXxXxXxXxXxXxXxXxXxXxXxXxX

STEP 2 — Get each caregiver's Chat ID (1 min each)
────────────────────────────────────────────────────
Each family member / caregiver does this:
1. Open Telegram → search your bot username → press START
2. Send any message (e.g. "hi")
3. Open this URL in a browser (replace TOKEN):
   https://api.telegram.org/bot<TOKEN>/getUpdates
4. Find the number next to "id" inside "chat":
   {"id": 987654321, "first_name": "Priya", ...}
   That number (987654321) is their Chat ID.

STEP 3 — Save to notify_config.json
──────────────────────────────────────
Go to MindBridge → Telegram Verification page → fill in:
  • Bot Token (from Step 1)
  • Patient name
  • Add each caregiver: name + chat ID + role
→ Click Save

STEP 4 — Test
─────────────
Click "Send test message" — each caregiver should receive a Telegram message.

VERIFICATION FLOW (automatic)
══════════════════════════════
1. Unknown person appears on camera
2. AI asks: "Sorry, may I know your name?"
3. Person says (or types) their name
4. App captures their photo + sends to ALL caregivers:

   ┌─────────────────────────────────────────────┐
   │ 🔔 Visitor verification needed              │
   │                                             │
   │ Someone named Skanda (says: Son) is with   │
   │ Grandma. Photo attached.                    │
   │                                             │
   │ Reply /yes to confirm → AI learns their    │
   │ face permanently.                           │
   │ Reply /no to reject → AI asks again.       │
   └─────────────────────────────────────────────┘

5. Any caregiver replies /yes  → AI trains FSL, welcomes person
   Any caregiver replies /no   → AI resets, asks name again
"""

import json
import os
import time
import requests


class TelegramNotifier:

    def __init__(self, config_path: str):
        self.config_path      = config_path
        self.config           = {}
        self.enabled          = False
        self._last_sent       = {}
        self._last_update_id  = 0
        self.reload_config()

    def reload_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path) as f:
                self.config = json.load(f)
        tok  = self.config.get("bot_token", "").strip()
        cgs  = self.config.get("caregivers", [])
        self.enabled = bool(tok and cgs)
        self._base   = f"https://api.telegram.org/bot{tok}"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _send_text(self, chat_id: str, text: str):
        try:
            requests.post(
                f"{self._base}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as e:
            print(f"[TG] sendMessage failed: {e}")

    def _send_photo(self, chat_id: str, image_path: str, caption: str = ""):
        try:
            with open(image_path, "rb") as img:
                requests.post(
                    f"{self._base}/sendPhoto",
                    data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                    files={"photo": img},
                    timeout=20,
                )
        except Exception as e:
            print(f"[TG] sendPhoto failed: {e}")

    def _broadcast(self, text: str, image_path: str = None):
        """Send text or photo+caption to all configured caregivers."""
        for cg in self.config.get("caregivers", []):
            cid = str(cg.get("chat_id", "")).strip()
            if not cid:
                continue
            if image_path and os.path.exists(image_path):
                self._send_photo(cid, image_path, caption=text)
            else:
                self._send_text(cid, text)

    # ── Public API ────────────────────────────────────────────────────────────

    def send_new_person_request(self,
                                 name: str,
                                 relation_hint: str,
                                 image_path: str,
                                 patient_name: str):
        """
        Called when an unknown person says their name.
        Sends photo + verification request to all caregivers.
        """
        rel_str = f" (says they are the {relation_hint})" if relation_hint else ""
        msg = (
            f"🔔 <b>Visitor verification needed</b>\n\n"
            f"Someone named <b>{name}</b>{rel_str} "
            f"is with <b>{patient_name}</b>.\n\n"
            f"• Reply <b>/yes</b> → AI learns their face permanently\n"
            f"• Reply <b>/no</b>  → AI resets and asks again"
        )
        self._broadcast(msg, image_path)

    def send_unknown_alert(self, image_path: str = None, conf: float = 0.0):
        """
        Called when face is seen but no name given.
        Respects cooldown to avoid spam.
        """
        cd   = self.config.get("unknown_cooldown_min", 3) * 60
        last = self._last_sent.get("unknown", 0)
        if time.time() - last < cd:
            return
        self._last_sent["unknown"] = time.time()

        patient = self.config.get("patient_name", "the patient")
        msg = (
            f"🚨 <b>Unknown person detected</b>\n\n"
            f"An unrecognised person is with <b>{patient}</b>.\n"
            f"Best match score: <b>{conf*100:.0f}%</b> (below threshold).\n"
            f"Please check immediately."
        )
        self._broadcast(msg, image_path)

    def send_alert(self, message: str, image_path: str = None):
        """Generic alert broadcast."""
        self._broadcast(message, image_path)

    def poll_response(self, expected_name: str) -> str | None:
        """
        Poll Telegram getUpdates for /yes or /no from any caregiver.
        Returns "yes", "no", or None.

        Uses offset so each update is only seen once.
        """
        try:
            r = requests.get(
                f"{self._base}/getUpdates",
                params={
                    "offset":  self._last_update_id + 1,
                    "timeout": 2,
                    "limit":   20,
                },
                timeout=6,
            ).json()

            for upd in r.get("result", []):
                uid = upd.get("update_id", 0)
                if uid > self._last_update_id:
                    self._last_update_id = uid

                txt = upd.get("message", {}).get("text", "").strip().lower()
                if txt == "/yes":
                    return "yes"
                if txt == "/no":
                    return "no"

        except Exception as e:
            print(f"[TG] poll_response error: {e}")

        return None

    def test_connection(self) -> tuple[bool, str]:
        """Verify bot token and send a test message to all caregivers."""
        try:
            r = requests.get(f"{self._base}/getMe", timeout=6).json()
            if r.get("ok"):
                bot_name = r["result"]["username"]
                self.send_alert(
                    f"✅ <b>MindBridge AI connected</b>\n\n"
                    f"Bot: @{bot_name}\n"
                    f"Patient: {self.config.get('patient_name','—')}\n"
                    f"Caregivers: {len(self.config.get('caregivers',[]))}\n\n"
                    f"You will receive verification requests here."
                )
                return True, f"Connected as @{bot_name} — test message sent."
            return False, f"Telegram error: {r}"
        except Exception as e:
            return False, f"Connection failed: {e}"