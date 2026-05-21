"""Tests for SMTP helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.config import Settings, get_settings
from app.mail.smtp_client import send_plain_text_email, smtp_is_configured


def test_smtp_is_configured_requires_credentials() -> None:
    cfg = Settings(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="",
        smtp_pass="",
    )
    assert smtp_is_configured(cfg) is False

    cfg_ok = Settings(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="user@example.com",
        smtp_pass="secret",
    )
    assert smtp_is_configured(cfg_ok) is True


@patch("app.mail.smtp_client.smtplib.SMTP")
def test_send_plain_text_email_uses_starttls_and_login(mock_smtp_class: MagicMock) -> None:
    mock_server = MagicMock()
    mock_smtp_class.return_value.__enter__.return_value = mock_server

    cfg = Settings(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="from@example.com",
        smtp_pass="secret",
        smtp_use_tls=True,
    )

    send_plain_text_email(
        to_addr="to@example.com",
        subject="Hello",
        body="Body text",
        settings=cfg,
    )

    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once_with("from@example.com", "secret")
    mock_server.send_message.assert_called_once()
