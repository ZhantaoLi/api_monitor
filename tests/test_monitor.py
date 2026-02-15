"""Tests for monitor.py – protocol routing and utility functions."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from monitor import (
    MonitorService,
    normalize_base_url,
    _check_response_body_for_error,
    _extract_text_from_chat,
    _extract_text_from_anthropic,
    _extract_text_from_gemini,
    _extract_text_from_responses,
)


class TestNormalizeBaseUrl:
    def test_strip_trailing_slash(self):
        assert normalize_base_url("https://api.example.com/") == "https://api.example.com"

    def test_strip_v1(self):
        assert normalize_base_url("https://api.example.com/v1") == "https://api.example.com"

    def test_strip_v1_trailing_slash(self):
        assert normalize_base_url("https://api.example.com/v1/") == "https://api.example.com"

    def test_no_change(self):
        assert normalize_base_url("https://api.example.com") == "https://api.example.com"

    def test_strip_whitespace(self):
        assert normalize_base_url("  https://api.example.com  ") == "https://api.example.com"

    def test_empty_string(self):
        assert normalize_base_url("") == ""


class TestCheckResponseBodyForError:
    def test_no_error(self):
        assert _check_response_body_for_error({"data": []}) is None

    def test_error_string(self):
        assert _check_response_body_for_error({"error": "oops"}) == "oops"

    def test_error_dict(self):
        result = _check_response_body_for_error({"error": {"message": "bad request"}})
        assert result == "bad request"

    def test_success_false(self):
        result = _check_response_body_for_error({"success": False, "message": "fail"})
        assert result == "fail"

    def test_non_dict(self):
        assert _check_response_body_for_error("string") is None
        assert _check_response_body_for_error(None) is None


class TestExtractTextFromChat:
    def test_standard_response(self):
        body = {
            "choices": [
                {"message": {"content": "Hello world"}}
            ]
        }
        assert _extract_text_from_chat(body) == "Hello world"

    def test_empty_choices(self):
        assert _extract_text_from_chat({"choices": []}) is None

    def test_no_choices(self):
        assert _extract_text_from_chat({}) is None


class TestExtractTextFromAnthropic:
    def test_standard_response(self):
        body = {
            "content": [
                {"type": "text", "text": "Hello from Claude"}
            ]
        }
        assert _extract_text_from_anthropic(body) == "Hello from Claude"

    def test_empty_content(self):
        assert _extract_text_from_anthropic({"content": []}) is None


class TestExtractTextFromGemini:
    def test_standard_response(self):
        body = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini says hi"}]
                    }
                }
            ]
        }
        assert _extract_text_from_gemini(body) == "Gemini says hi"

    def test_empty_candidates(self):
        assert _extract_text_from_gemini({"candidates": []}) is None


class TestExtractTextFromResponses:
    def test_standard_response(self):
        body = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "Response text"}
                    ]
                }
            ]
        }
        assert _extract_text_from_responses(body) == "Response text"


class TestChooseRoute:
    """Test the protocol routing rules."""

    @pytest.fixture()
    def service(self, tmp_path):
        from db import Database
        d = Database(str(tmp_path / "t.db"))
        d.init_db()
        return MonitorService(db=d, log_dir=str(tmp_path / "logs"))

    def test_claude_route(self, service):
        assert service._choose_route("claude-3-opus") == "anthropic"

    def test_gemini_route(self, service):
        assert service._choose_route("gemini-pro") == "gemini"

    def test_gpt_chat_route(self, service):
        assert service._choose_route("gpt-4o") == "chat"

    def test_codex_responses_route(self, service):
        assert service._choose_route("codex-mini") == "responses"

    def test_default_chat(self, service):
        assert service._choose_route("some-random-model") == "chat"

    def test_gpt5_responses_route(self, service):
        assert service._choose_route("gpt-5.1") == "responses"
