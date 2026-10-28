"""Tests for trial quota limits, developer overrides, and audit log."""

import pytest
from backend.storage import (
    get_user_trial_status,
    check_and_increment_trial_quota,
    grant_user_trial_runs,
    reset_user_trials,
    get_trial_audit_log,
    get_current_system_notice,
    USER_TRIALS_FILE,
    TRIAL_AUDIT_LOG_FILE,
    USER_NOTICES_FILE,
    _write_json,
)


@pytest.fixture(autouse=True)
def clean_trials_state(tmp_path, monkeypatch):
    """Isolate trial metadata files for each test."""
    trials_f = tmp_path / "user_trials.json"
    audit_f = tmp_path / "trial_audit_log.json"
    notices_f = tmp_path / "user_notices.json"

    monkeypatch.setattr("backend.storage.USER_TRIALS_FILE", trials_f)
    monkeypatch.setattr("backend.storage.TRIAL_AUDIT_LOG_FILE", audit_f)
    monkeypatch.setattr("backend.storage.USER_NOTICES_FILE", notices_f)


def test_trial_quota_defaults():
    status = get_user_trial_status("test_u1")
    assert status["guided_used"] == 0
    assert status["guided_limit"] == 2
    assert status["guided_remaining"] == 2
    assert status["expert_used"] == 0
    assert status["expert_limit"] == 1
    assert status["expert_remaining"] == 1


def test_guided_mode_quota_limit():
    user_id = "test_guided_user"

    # Run 1 - should succeed
    res1 = check_and_increment_trial_quota(user_id, mode="guided", is_admin=False)
    assert res1["status"] == "allowed"
    assert res1["guided_used"] == 1

    # Run 2 - should succeed
    res2 = check_and_increment_trial_quota(user_id, mode="guided", is_admin=False)
    assert res2["status"] == "allowed"
    assert res2["guided_used"] == 2

    # Run 3 - should fail (max 2 runs)
    with pytest.raises(RuntimeError) as exc_info:
        check_and_increment_trial_quota(user_id, mode="guided", is_admin=False)
    assert "You have used all 2 free Guided mode experiments" in str(exc_info.value)


def test_expert_mode_quota_limit():
    user_id = "test_expert_user"

    # Run 1 - should succeed
    res1 = check_and_increment_trial_quota(user_id, mode="expert", is_admin=False)
    assert res1["status"] == "allowed"
    assert res1["expert_used"] == 1

    # Run 2 - should fail (max 1 run)
    with pytest.raises(RuntimeError) as exc_info:
        check_and_increment_trial_quota(user_id, mode="expert", is_admin=False)
    assert "You have used your free Expert mode experiment" in str(exc_info.value)


def test_admin_developer_unlimited_bypass():
    admin_id = "founder_rishi"

    # Developer can run 10 times without error
    for _ in range(10):
        res_g = check_and_increment_trial_quota(admin_id, mode="guided", is_admin=True)
        assert res_g["status"] == "unlimited"
        res_e = check_and_increment_trial_quota(admin_id, mode="expert", is_admin=True)
        assert res_e["status"] == "unlimited"

    status = get_user_trial_status(admin_id, is_admin=True)
    assert status["is_admin"] is True
    assert status["guided_remaining"] > 1000


def test_grant_user_trial_runs_and_notice():
    user_id = "student_u1"

    # Exhaust guided quota
    check_and_increment_trial_quota(user_id, mode="guided", is_admin=False)
    check_and_increment_trial_quota(user_id, mode="guided", is_admin=False)

    # Verify blocked
    with pytest.raises(RuntimeError):
        check_and_increment_trial_quota(user_id, mode="guided", is_admin=False)

    # Developer grants +3 Guided and +2 Expert runs
    grant_res = grant_user_trial_runs(
        user_id=user_id,
        guided_runs=3,
        expert_runs=2,
        user_email="student@college.edu",
        adjusted_by="Developer: Adepu Rishi",
    )
    assert grant_res["success"] is True
    assert grant_res["guided_limit"] == 5  # 2 base + 3 bonus

    # Now run 3 should succeed
    res3 = check_and_increment_trial_quota(user_id, mode="guided", is_admin=False)
    assert res3["status"] == "allowed"
    assert res3["guided_used"] == 3

    # Notice was created for user
    notice = get_current_system_notice(user_id)
    assert notice is not None
    assert "Extra Free Trials Granted" in notice["title"]
    assert "URQEN has granted" in notice["message"]

    # Audit log was recorded
    audit = get_trial_audit_log()
    assert len(audit) >= 1
    assert audit[0]["action"] == "GRANTED_BONUS_RUNS"
    assert audit[0]["guided_bonus"] == 3
    assert audit[0]["adjusted_by"] == "Developer: Adepu Rishi"


def test_reset_user_trials_and_notice():
    user_id = "student_u2"

    # Use expert run
    check_and_increment_trial_quota(user_id, mode="expert", is_admin=False)
    with pytest.raises(RuntimeError):
        check_and_increment_trial_quota(user_id, mode="expert", is_admin=False)

    # Developer resets trials
    reset_res = reset_user_trials(user_id=user_id, user_email="student2@college.edu")
    assert reset_res["success"] is True
    assert reset_res["expert_used"] == 0

    # User can now run again
    res = check_and_increment_trial_quota(user_id, mode="expert", is_admin=False)
    assert res["status"] == "allowed"
    assert res["expert_used"] == 1

    # Notice created
    notice = get_current_system_notice(user_id)
    assert notice is not None
    assert "Free Trials Reset" in notice["title"]
