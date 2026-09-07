import time

from ser_lib.service.jobs import JobManager


def _wait_terminal(manager, job_id, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if record["status"] in {"completed", "failed", "cancelled"}:
            return record
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_job_manager_reports_progress_and_result():
    manager = JobManager(max_workers=1)
    try:
        job_id = manager.submit("demo", lambda context: (context.report(0.5, "half"), 7)[1])
        record = _wait_terminal(manager, job_id)
        assert record["status"] == "completed"
        assert record["progress"] == 1.0
        assert record["result"] == 7
    finally:
        manager.shutdown()


def test_job_manager_records_failure():
    manager = JobManager(max_workers=1)
    try:
        def fail(context):
            raise RuntimeError("boom")
        record = _wait_terminal(manager, manager.submit("demo", fail))
        assert record["status"] == "failed"
        assert "boom" in record["error"]
    finally:
        manager.shutdown()
