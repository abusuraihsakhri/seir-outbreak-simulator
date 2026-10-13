"""Security and CLI regression tests."""
import csv
from pathlib import Path

from agents.base import AuditTrail
from cli import main


def test_hmac_verification_rejects_modified_entries():
    trail = AuditTrail(secret_key="test-only-in-memory-key")
    trail.log("tester", "unit", "event", {"status": "OK"})
    assert trail.verify_integrity()
    trail.logs[0]["event_type"] = "OTHER"
    assert not trail.verify_integrity()


def test_hmac_verification_rejects_forged_link_even_with_matching_pointer():
    trail = AuditTrail(secret_key="test-only-in-memory-key")
    trail.log("tester", "unit", "first", {"status": "OK"})
    trail.log("tester", "unit", "second", {"status": "OK"})
    trail.logs[0]["current_hash"] = "0" * 64
    trail.logs[1]["prev_hash"] = "0" * 64
    assert not trail.verify_integrity()


def test_public_audit_trail_is_defensive_copy():
    trail = AuditTrail(secret_key="test-only-in-memory-key")
    trail.log("tester", "unit", "event", {"status": "OK"})
    result = trail.get_trail()
    result[0]["actor"] = "forged"
    assert trail.verify_integrity()
    assert trail.get_trail()[0]["actor"] == "tester"


def test_batch_boolean_false_and_true_are_distinct(tmp_path, capsys):
    source = tmp_path / "input.csv"
    target = tmp_path / "output.csv"
    with source.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "target_identifier",
                                                "primary_metric", "secondary_metric",
                                                "is_critical_flag"])
        writer.writeheader()
        for value, key in [("False", "CASE-FALSE"), ("True", "CASE-TRUE")]:
            writer.writerow(dict(task_id=key, target_identifier=key,
                                 primary_metric=1, secondary_metric=1,
                                 is_critical_flag=value))
    assert main(["batch", "-i", str(source), "-o", str(target)]) == 0
    with target.open(newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["overall_urgency"] == "ROUTINE"
    assert rows[1]["overall_urgency"] == "CRITICAL_STAT_PANIC"


def test_hmac_signature_cannot_be_rewritten_by_moving_delimiters():
    trail = AuditTrail(secret_key="test-only-in-memory-key")
    trail.log("alice|admin", "tier", "event", {"status": "OK"})
    assert trail.verify_integrity()
    trail.logs[0]["actor"] = "alice"
    trail.logs[0]["actor_tier"] = "admin|tier"
    assert not trail.verify_integrity()
