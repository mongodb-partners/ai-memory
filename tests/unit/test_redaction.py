"""Tests for audit-record error redaction. REQ-E-084.

The threat is not an attacker; it is a careless library. Driver and provider
exceptions quote what they failed on, and `str(exc)` in an audit record publishes
that quote into a collection retained for weeks and readable by anyone with the
admin role — a wider and longer-lived audience than the process log.
"""

from agent_memory.core.redaction import redact_error, redact_message


class TestConnectionStrings:
    """The leak that matters most: an SRV URI carries its password inline."""

    def test_a_mongodb_srv_password_is_removed(self):
        """TC-RED-001: this one writes cluster credentials into the cluster.

        A `ServerSelectionTimeoutError` names the topology it could not reach, and
        for `mongodb+srv://user:password@host` that name contains the password. The
        audit collection lives in the same database the URI authenticates to, on
        the admin-readable side of the system.
        """
        out = redact_message(
            "connection to mongodb+srv://svc_user:Tr0ub4dor&3@c0.abc.mongodb.net "
            "timed out"
        )
        assert "Tr0ub4dor&3" not in out
        # The principal survives: "which credential failed" is the diagnosis.
        assert "svc_user" in out
        assert "c0.abc.mongodb.net" in out

    def test_a_plain_mongodb_uri_is_covered_too(self):
        # TC-RED-002
        assert "hunter2" not in redact_message(
            "auth failed for mongodb://admin:hunter2@10.0.0.4:27017/agent_memory"
        )

    def test_a_uri_without_credentials_is_left_alone(self):
        # TC-RED-003: no false positives on the common local case.
        text = "connection to mongodb://localhost:27017 refused"
        assert redact_message(text) == text


class TestProviderSecrets:
    """Embedding and LLM SDKs echo the request that failed."""

    def test_a_bearer_token_is_removed(self):
        # TC-RED-010
        out = redact_message(
            "401 Unauthorized: {'Authorization': 'Bearer sk-ant-api03-AAAABBBBCCCC'}"
        )
        assert "sk-ant-api03-AAAABBBBCCCC" not in out

    def test_an_assigned_secret_is_removed(self):
        # TC-RED-011
        for text in (
            "invalid request api_key=pa-abcdefghijklmnop",
            'bad config {"password": "letmein123"}',
            "VOYAGE_API_KEY: pa-QQQQWWWWEEEE rejected",
        ):
            out = redact_message(text)
            assert "abcdefghijklmnop" not in out
            assert "letmein123" not in out
            assert "QQQQWWWWEEEE" not in out

    def test_a_prefixed_key_is_removed_even_when_unlabelled(self):
        """TC-RED-012: some keys are secrets by shape, with no `key=` to match."""
        out = redact_message("signature mismatch for AKIAIOSFODNN7EXAMPLE in us-east-1")
        assert "AKIAIOSFODNN7EXAMPLE" not in out
        assert "us-east-1" in out


class TestShape:
    def test_the_exception_type_is_always_kept(self):
        """TC-RED-020: the type is the diagnosis and can never hold a secret.

        `ServerSelectionTimeoutError` vs `DuplicateKeyError` vs `ValueError` is
        what makes an audit entry actionable. Redaction that dropped it would trade
        a leak for a useless record.
        """
        assert redact_error(ValueError("nothing sensitive")).startswith("ValueError:")

    def test_a_long_message_is_capped(self):
        """TC-RED-021: a driver dumping a full topology description must not become
        the largest field in the audit collection."""
        out = redact_message("x" * 5000)
        assert len(out) < 400
        assert out.endswith("…")

    def test_an_empty_message_yields_just_the_type(self):
        # TC-RED-022
        assert redact_error(RuntimeError()) == "RuntimeError"

    def test_a_broken_dunder_str_still_yields_the_type(self):
        """TC-RED-023: redaction runs on the failure path and must not add one."""

        class Hostile(Exception):
            def __str__(self):
                raise RuntimeError("no")

        assert redact_error(Hostile()) == "Hostile"

    def test_an_ordinary_message_survives_intact(self):
        """TC-RED-024: redaction must not make normal errors unreadable.

        Over-aggressive scrubbing is its own failure: an audit trail of
        `[redacted]` tells an operator nothing and trains them to ignore the field.
        """
        text = "embedding dimension 1024 does not match index numDimensions 1536"
        assert redact_message(text) == text
