"""SMTP send helper using stdlib only."""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from app.config import Settings, get_settings
from app.observability.logging import log_event

logger = logging.getLogger("app.mail.smtp_client")


def smtp_is_configured(settings: Settings | None = None) -> bool:
    cfg = settings or get_settings()
    return bool(cfg.smtp_host and cfg.smtp_port and cfg.smtp_user.strip() and cfg.smtp_pass.strip())


def send_plain_text_email(
    *,
    to_addr: str,
    subject: str,
    body: str,
    settings: Settings | None = None,
) -> None:
    """Send one plain-text email using SMTP STARTTLS (typical for port 587)."""

    cfg = settings or get_settings()
    if not smtp_is_configured(cfg):
        raise RuntimeError("SMTP is not configured (set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS).")

    sender = (cfg.smtp_from or cfg.smtp_user).strip()
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_addr.strip()
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=cfg.smtp_timeout_seconds) as server:
        server.ehlo()
        if cfg.smtp_use_tls:
            server.starttls(context=context)
            server.ehlo()
        server.login(cfg.smtp_user, cfg.smtp_pass)
        server.send_message(msg)

    log_event(logger, event="smtp.sent", to=to_addr, subject=subject)
