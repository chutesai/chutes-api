import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

import api.database.orms  # noqa: F401
import api.external_backend.polling as polling
from api.external_backend.operation_lifecycle import UsageCheckpointLoop
from api.external_backend.polling import (
    AccountSnapshot,
    EndpointRequest,
    ExternalOperationPoller,
    LeasedOperation,
    PollingConfigurationError,
    PollOutcome,
    TaskLifecyclePolicy,
    WorkerSettings,
    build_claim_statement,
    build_governance_bucket_prune_statement,
    build_missing_governance_scope_statement,
    build_settlement_reconcile_statement,
    build_usage_outbox_reconcile_statement,
)
from api.external_backend.public_urls import artifact_url
from api.external_backend.schemas import ExternalRouteConfig
from api.external_transport import BodyMode, BufferedResponse
from api.payment.pricing import NormalizedUsage


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def route(operation_config, *, response_config=None, operation_mode="task"):
    return ExternalRouteConfig.model_validate(
        {
            "cord_path": "/generate",
            "upstream_resource_id": "resource-id",
            "operation_mode": operation_mode,
            "protocol": "generic-json",
            "path_template": "/tasks",
            "method": "POST",
            "request_config": {"body_mode": "json"},
            "response_config": response_config or {},
            "operation_config": operation_config,
            "capabilities": {},
        }
    )


def account():
    return AccountSnapshot(
        account_id="account-id",
        user_id="user-id",
        base_url="https://service.example.test",
        credential_references={},
        auth_header_templates=(),
        connection_config={},
    )


def lease(route_config, *, attempts=0, usage=None, status="submitted"):
    return LeasedOperation(
        operation_id="operation-id",
        lease_token="worker:lease",
        status=status,
        poll_attempts=attempts,
        task_id="remote-task-id",
        route=route_config,
        account=account(),
        request_metadata={"quality": "high"},
        upstream_metadata={},
        usage=usage,
        result_descriptor=None,
        expires_at=NOW + timedelta(hours=1),
    )


def poll_config(**overrides):
    value = {
        "endpoint": {
            "path_template": "/tasks/{task_id}",
            "method": "GET",
        },
        "task": {
            "status": {
                "path": "state",
                "map": {
                    "WAITING": "pending",
                    "WORKING": "running",
                    "COMPLETE": "succeeded",
                    "ERROR": "failed",
                },
                "required": True,
            },
            "result": "output",
            "artifacts": {
                "items": "output.files[*]",
                "url": {"path": "url", "required": True},
                "kind": {"value": "video"},
                "content_type": "mime",
                "size_bytes": "size",
            },
        },
        "usage": {
            "default_requests": 1,
            "fields": {"tokens.output": "usage.output_tokens"},
        },
        "interval_seconds": 3,
        "backoff": {"multiplier": 2, "maximum_seconds": 30, "jitter_fraction": 0},
        "retry": {"statuses": [429, 503], "max_attempts": 10},
    }
    value.update(overrides)
    return value


class FakeExecutor:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def execute(self, profile, outbound):
        self.calls.append((profile, outbound))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def response(status, body, *, private_headers=None):
    return BufferedResponse(
        status_code=status,
        headers={"content-type": "application/json"},
        body=orjson.dumps(body),
        private_headers=private_headers or {},
    )


def make_poller(executor, *, random_value=lambda: 0.5):
    async def no_settlement(_event):
        return None

    return ExternalOperationPoller(
        executor_factory=lambda _account: executor,
        settlement_hook=no_settlement,
        clock=lambda: NOW,
        random_value=random_value,
    )


def test_task_lifecycle_policy_supports_top_level_and_nested_aliases():
    top_level = TaskLifecyclePolicy.from_route(
        route(
            {
                "poll": poll_config(),
                "cancel": {
                    "endpoint": {
                        "path_template": "/tasks/{task_id}",
                        "method": "DELETE",
                    }
                },
            }
        )
    )
    nested = TaskLifecyclePolicy.from_route(route({"task": {"poll": poll_config()}}))

    assert top_level.poll.task is not None
    assert top_level.cancel is not None
    assert nested.poll.task is not None
    assert top_level.poll.endpoint["body_mode"] == "none"
    assert top_level.poll.endpoint["response_mode"] == "buffered"


def test_settlement_reconcile_selects_pending_and_failed_terminal_rows():
    statement = build_settlement_reconcile_statement(NOW, 37)
    criteria = str(statement.whereclause).upper()
    collection_parameters = [
        set(value)
        for value in statement.compile().params.values()
        if isinstance(value, (set, frozenset, list, tuple))
    ]

    assert {"pending", "failed"} in collection_parameters
    assert {
        "succeeded",
        "failed",
        "cancelled",
        "expired",
    } in collection_parameters
    assert statement._for_update_arg.skip_locked is True
    assert [table.name for table in statement._for_update_arg.of] == [
        "external_operations"
    ]
    assert "NEXT_POLL_AT IS NULL" in criteria
    assert "NEXT_POLL_AT <=" in criteria
    assert "NOT (EXISTS" in criteria
    assert statement._limit_clause.value == 37


def test_usage_outbox_reconcile_uses_its_due_timestamp_and_skip_locked():
    statement = build_usage_outbox_reconcile_statement(NOW, 19)
    criteria = str(statement.whereclause).upper()

    assert "NEXT_ATTEMPT_AT IS NULL" in criteria
    assert "NEXT_ATTEMPT_AT <=" in criteria
    assert statement._for_update_arg.skip_locked is True
    assert [table.name for table in statement._for_update_arg.of] == [
        "external_usage_outbox"
    ]
    assert statement._limit_clause.value == 19


@pytest.mark.asyncio
async def test_settlement_reconcile_claim_is_atomic_across_workers(monkeypatch):
    operation = SimpleNamespace(
        operation_id="operation-id",
        next_poll_at=None,
    )
    transaction_lock = asyncio.Lock()
    settlement_calls = []

    class Result:
        def __init__(self, rows):
            self._rows = rows

        def unique(self):
            return self

        def scalars(self):
            return self

        def all(self):
            return self._rows

    class Transaction:
        async def __aenter__(self):
            await transaction_lock.acquire()

        async def __aexit__(self, *_args):
            transaction_lock.release()

    execute_count = 0

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def begin(self):
            return Transaction()

        async def execute(self, _statement):
            nonlocal execute_count
            execute_count += 1
            # Each worker checks the outbox first; this fixture exercises the
            # pre-price operation queue which has no immutable event yet.
            if execute_count in {1, 3}:
                return Result([])
            due = operation.next_poll_at is None or operation.next_poll_at <= NOW
            return Result([operation] if due else [])

    async def settle(operation_id):
        settlement_calls.append(operation_id)

    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)
    settings = WorkerSettings(lease_seconds=60)
    workers = [
        ExternalOperationPoller(
            session_factory=Session,
            settings=settings,
            worker_id=f"worker-{index}",
            clock=lambda: NOW,
        )
        for index in range(2)
    ]

    counts = await asyncio.gather(
        *(worker.reconcile_settlements_once() for worker in workers)
    )

    assert sorted(counts) == [0, 1]
    assert settlement_calls == ["operation-id"]
    assert operation.next_poll_at == NOW + timedelta(seconds=60)


@pytest.mark.asyncio
async def test_due_outbox_is_reconciled_even_when_operation_state_is_stale(monkeypatch):
    event = SimpleNamespace(
        operation_id="operation-id",
        next_attempt_at=NOW,
    )
    execute_count = 0

    class Result:
        def __init__(self, rows):
            self._rows = rows

        def unique(self):
            return self

        def scalars(self):
            return self

        def all(self):
            return self._rows

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def begin(self):
            return self

        async def execute(self, _statement):
            nonlocal execute_count
            execute_count += 1
            return Result([event] if execute_count == 1 else [])

    settlement_calls = []

    async def settle(operation_id):
        settlement_calls.append(operation_id)

    monkeypatch.setattr("api.external_backend.service.settle_operation", settle)
    poller = ExternalOperationPoller(
        session_factory=Session,
        settings=WorkerSettings(lease_seconds=60),
        clock=lambda: NOW,
    )

    count = await poller.reconcile_settlements_once()

    assert count == 1
    assert settlement_calls == ["operation-id"]
    assert event.next_attempt_at == NOW + timedelta(seconds=60)


def test_top_level_poll_precedes_nested_alias():
    configured = route(
        {
            "poll": poll_config(interval_seconds=7),
            "task": {"poll": poll_config(interval_seconds=19)},
        }
    )

    assert TaskLifecyclePolicy.from_route(configured).poll.backoff.interval_seconds == 7


@pytest.mark.parametrize(
    "operation_config",
    [
        {},
        {"poll": {"task": {"status": "state"}}},
        {"poll": {**poll_config(), "unknown": True}},
        {
            "poll": {
                **poll_config(),
                "endpoint": {"path_template": "/tasks", "unknown": True},
            }
        },
        {"poll": {**poll_config(), "retry": {"statuses": [99]}}},
        {
            "poll": {
                **poll_config(),
                "usage_mode": "delta",
                "usage": {"fields": {"tokens.output": "usage.output"}},
            }
        },
    ],
)
def test_polling_configuration_fails_closed(operation_config):
    with pytest.raises(PollingConfigurationError):
        TaskLifecyclePolicy.from_route(route(operation_config))


def test_delta_usage_requires_zero_observation_request_default():
    configured = poll_config(
        usage_mode="delta",
        usage={
            "default_requests": 0,
            "fields": {"tokens.output": "usage.output_tokens"},
        },
    )

    policy = TaskLifecyclePolicy.from_route(route({"poll": configured}))

    assert policy.poll.usage_mode.value == "delta"
    assert policy.poll.usage.default_requests == 0


def test_endpoint_request_maps_context_without_forwarding_client_headers():
    request = EndpointRequest.from_config(
        {
            "path_parameters": {"account": {"source": "context", "path": "account"}},
            "query": {
                "cursor": {"source": "context", "path": "cursor"},
                "missing": {"source": "context", "path": "missing"},
            },
            "body": {},
            "body_transform": {
                "inject": [
                    {
                        "target": "task",
                        "source": "context",
                        "path": "task_id",
                        "required": True,
                    }
                ]
            },
        },
        name="request",
    )
    profile = SimpleNamespace(
        path_template="/accounts/{account}/tasks/{task_id}", body_mode=BodyMode.JSON
    )
    sources = {
        "context": {
            "account": "account-id",
            "task_id": "task-id",
            "cursor": "next",
        },
        "request": {"authorization": "must-not-pass"},
        "response": None,
    }

    outbound = request.build(profile, sources)

    assert dict(outbound.path_parameters) == {
        "account": "account-id",
        "task_id": "task-id",
    }
    assert dict(outbound.query) == {"cursor": "next"}
    assert dict(outbound.headers) == {}
    assert outbound.body.value == {"task": "task-id"}


@pytest.mark.parametrize("name", ["signature", "vendor_signature", "x-api-key"])
def test_endpoint_request_rejects_plaintext_credential_query_configuration(name):
    with pytest.raises(PollingConfigurationError, match="credential-like"):
        EndpointRequest.from_config(
            {"query": {name: {"value": "plain-text-secret"}}},
            name="request",
        )


def test_poll_success_extracts_usage_and_only_persists_remote_artifact_metadata():
    remote_url = "https://objects.example.test/result.mp4"
    executor = FakeExecutor(
        response(
            200,
            {
                "state": "COMPLETE",
                "usage": {"output_tokens": 12},
                "output": {
                    "id": "remote-task-id",
                    "model": "private-model",
                    "provider_name": "private-name",
                    "files": [
                        {
                            "url": remote_url,
                            "mime": "video/mp4",
                            "size": 1024,
                        }
                    ],
                },
            },
        )
    )
    route_config = route({"persist_inline_result": True, "poll": poll_config()})
    poller = make_poller(executor)

    outcome = asyncio.run(poller._poll(lease(route_config)))

    assert outcome.status == "succeeded"
    assert outcome.billable is True
    assert outcome.usage.requests == 1
    assert outcome.usage.tokens == {"output": 12}
    descriptor = outcome.result_descriptor
    public_artifact_url = artifact_url("operation-id", 0)
    assert descriptor["status"] == "complete"
    assert descriptor["artifacts"] == [
        {
            "kind": "video",
            "reference": remote_url,
            "content_type": "video/mp4",
            "size_bytes": 1024,
            "expires_at": (NOW + timedelta(days=1)).isoformat(),
            "attributes": {"local_path": public_artifact_url},
        }
    ]
    inline = descriptor["metadata"]["inline_result"]
    assert inline == {
        "id": "operation-id",
        "files": [
            {
                "url": public_artifact_url,
                "mime": "video/mp4",
                "size": 1024,
            }
        ],
    }
    assert not any("blob" in key for key in descriptor)
    profile, outbound = executor.calls[0]
    assert profile.method == "GET"
    assert profile.body_mode is BodyMode.NONE
    assert dict(outbound.path_parameters) == {"task_id": "remote-task-id"}
    assert dict(outbound.headers) == {}


def test_poll_does_not_replay_request_usage_from_reduced_metadata():
    configured = poll_config(
        usage_mode="delta",
        usage={
            "default_requests": 0,
            "fields": {
                "counts.requested_duration": {
                    "source": "request",
                    "path": "parameters.duration",
                    "required": True,
                },
                "output_media_seconds.output": {
                    "source": "response",
                    "path": "usage.duration",
                    "required": True,
                },
            },
        },
    )
    route_config = route({"poll": configured})
    executor = FakeExecutor(
        response(
            200,
            {
                "state": "COMPLETE",
                "usage": {"duration": 10},
                "output": {},
            },
        )
    )
    initial = NormalizedUsage.from_mapping(
        {"requests": 1, "counts": {"requested_duration": 5}}
    )

    outcome = asyncio.run(
        make_poller(executor)._poll(lease(route_config, usage=initial))
    )

    assert outcome.usage.requests == 1
    assert outcome.usage.counts == {"requested_duration": 5}
    assert outcome.usage.output_media_seconds == {"output": 10}


def test_pending_poll_is_normalized_to_submitted_and_backed_off():
    executor = FakeExecutor(
        response(200, {"state": "WAITING", "usage": {"output_tokens": 2}})
    )
    route_config = route({"poll": poll_config()})

    outcome = asyncio.run(make_poller(executor)._poll(lease(route_config, attempts=1)))

    assert outcome.status == "submitted"
    assert outcome.upstream_status == "pending"
    assert outcome.next_poll_at == NOW + timedelta(seconds=6)
    assert outcome.error is None


def test_poll_limit_and_retryable_http_status_are_configurable():
    limited_config = poll_config(retry={"statuses": [503], "max_attempts": 2})
    route_config = route({"poll": limited_config})
    retrying = asyncio.run(
        make_poller(FakeExecutor(response(503, {})))._poll(
            lease(route_config, attempts=0)
        )
    )
    exhausted = asyncio.run(
        make_poller(FakeExecutor(response(503, {})))._poll(
            lease(route_config, attempts=1)
        )
    )

    assert retrying.status == "submitted"
    assert retrying.next_poll_at is not None
    assert exhausted.status == "failed"
    assert exhausted.error["code"] == "poll_limit_exceeded"
    assert exhausted.billable is True


def test_retry_delay_uses_configured_private_control_header_without_public_forwarding():
    configured = poll_config(
        retry={
            "statuses": [503],
            "max_attempts": 4,
            "retry_after_headers": ["x-control-wait"],
        }
    )
    route_config = route({"poll": configured})
    executor = FakeExecutor(response(503, {}, private_headers={"x-control-wait": "9"}))

    outcome = asyncio.run(make_poller(executor)._poll(lease(route_config)))

    assert outcome.next_poll_at == NOW + timedelta(seconds=9)
    profile, _ = executor.calls[0]
    assert profile.private_response_headers == frozenset({"x-control-wait"})
    assert "x-control-wait" not in profile.allowed_response_headers


def test_billable_terminal_defaults_cover_accepted_failure_and_can_be_overridden():
    failed = response(200, {"state": "ERROR", "usage": {"output_tokens": 3}})
    default_route = route({"poll": poll_config()})
    free_route = route({"poll": poll_config(billable_statuses=[])})

    default = asyncio.run(make_poller(FakeExecutor(failed))._poll(lease(default_route)))
    overridden = asyncio.run(make_poller(FakeExecutor(failed))._poll(lease(free_route)))

    assert default.status == "failed"
    assert default.billable is True
    assert overridden.billable is False


def test_cancel_requested_operation_uses_cancel_mapping_and_marks_dispatch():
    configured = route(
        {
            "poll": poll_config(),
            "cancel": {
                "endpoint": {
                    "path_template": "/tasks/{task_id}",
                    "method": "DELETE",
                }
            },
        }
    )
    executor = FakeExecutor(response(204, {}))
    poller = make_poller(executor)
    requested = replace(
        lease(configured),
        cancel_requested=True,
        cancel_requested_at=NOW.isoformat(),
    )

    outcome = asyncio.run(poller._cancel_outcome(requested))

    assert outcome.status == "cancelled"
    assert outcome.billable is True
    assert outcome.cancel_dispatched is True
    profile, outbound = executor.calls[0]
    assert profile.method == "DELETE"
    assert dict(outbound.headers) == {}


def test_cancel_does_not_replay_request_usage_from_reduced_metadata():
    configured = route(
        {
            "poll": poll_config(),
            "cancel": {
                "endpoint": {
                    "path_template": "/tasks/{task_id}",
                    "method": "DELETE",
                },
                "task": {
                    "status": {
                        "path": "state",
                        "map": {"CANCELLED": "cancelled"},
                        "required": True,
                    }
                },
                "usage_mode": "delta",
                "usage": {
                    "default_requests": 0,
                    "fields": {
                        "counts.requested_duration": {
                            "source": "request",
                            "path": "parameters.duration",
                            "required": True,
                        },
                        "output_media_seconds.output": {
                            "source": "response",
                            "path": "usage.duration",
                            "required": True,
                        },
                    },
                },
            },
        }
    )
    executor = FakeExecutor(
        response(200, {"state": "CANCELLED", "usage": {"duration": 3}})
    )
    initial = NormalizedUsage.from_mapping(
        {"requests": 1, "counts": {"requested_duration": 5}}
    )

    outcome = asyncio.run(
        make_poller(executor)._cancel_outcome(lease(configured, usage=initial))
    )

    assert outcome.status == "cancelled"
    assert outcome.usage.requests == 1
    assert outcome.usage.counts == {"requested_duration": 5}
    assert outcome.usage.output_media_seconds == {"output": 3}


def test_process_lease_dispatches_persisted_cancel_request_before_polling():
    configured = route(
        {
            "poll": poll_config(),
            "cancel": {
                "endpoint": {
                    "path_template": "/tasks/{task_id}",
                    "method": "DELETE",
                }
            },
        }
    )
    executor = FakeExecutor(response(204, {}))

    class RecordingPoller(ExternalOperationPoller):
        async def _renew_lease(self, _lease, _lost):
            await asyncio.Event().wait()

        async def _poll(self, _lease):
            raise AssertionError("poll must not run before requested cancellation")

        async def _finalize(self, _lease, outcome):
            self.finalized = outcome
            return None

    async def no_settlement(_event):
        return None

    poller = RecordingPoller(
        executor_factory=lambda _account: executor,
        settlement_hook=no_settlement,
        clock=lambda: NOW,
    )
    requested = replace(lease(configured), cancel_requested=True)

    asyncio.run(poller._process_lease(requested))

    assert poller.finalized.status == "cancelled"
    assert poller.finalized.cancel_dispatched is True


def test_lease_snapshot_carries_persisted_cancellation_state():
    route_config = route({"poll": poll_config()})
    account_record = SimpleNamespace(
        account_id="account-id",
        user_id="user-id",
        base_url="https://service.example.test",
        credential_references={},
        auth_header_templates=[],
        connection_config={},
    )
    operation = SimpleNamespace(
        operation_id="operation-id",
        status="submitted",
        poll_attempts=0,
        upstream_operation_id="remote-task-id",
        route_snapshot=route_config.model_dump(mode="json"),
        account=account_record,
        request_metadata={},
        upstream_metadata={},
        usage=None,
        result_descriptor=None,
        expires_at=None,
        settlement_metadata={
            "cancel_requested": True,
            "cancel_requested_at": NOW.isoformat(),
            "cancel_dispatched": False,
        },
    )

    snapshot = polling._lease_snapshot(operation, "worker:lease")

    assert snapshot.cancel_requested is True
    assert snapshot.cancel_requested_at == NOW.isoformat()
    assert snapshot.cancel_dispatched is False


@pytest.mark.asyncio
async def test_finalize_wakes_fresh_cancel_request_instead_of_storing_backoff():
    configured = route({"poll": poll_config()})
    leased = lease(configured, status="submitted")
    operation = SimpleNamespace(
        operation_id=leased.operation_id,
        lease_owner=leased.lease_token,
        lease_expires_at=NOW + timedelta(seconds=60),
        status="submitted",
        poll_attempts=0,
        last_polled_at=None,
        upstream_status=None,
        usage=None,
        result_descriptor=None,
        error=None,
        next_poll_at=None,
        started_at=None,
        finished_at=None,
        settlement_metadata={
            "cancel_requested": True,
            "cancel_dispatched": False,
        },
    )

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def begin(self):
            return self

        async def get(self, *_args, **_kwargs):
            return operation

    poller = ExternalOperationPoller(session_factory=Session, clock=lambda: NOW)
    outcome = PollOutcome(
        status="running",
        upstream_status="working",
        usage=NormalizedUsage(requests=1),
        result_descriptor=None,
        error=None,
        next_poll_at=NOW + timedelta(seconds=30),
    )

    event = await poller._finalize(leased, outcome)

    assert event is None
    assert operation.next_poll_at == NOW
    assert operation.settlement_metadata["cancel_requested"] is True


def test_expired_accepted_operation_is_billable_without_network_request():
    executor = FakeExecutor(response(200, {"state": "COMPLETE"}))
    route_config = route({"poll": poll_config()})
    expired = lease(route_config)
    expired = replace(expired, expires_at=NOW - timedelta(seconds=1))

    outcome = asyncio.run(make_poller(executor)._poll(expired))

    assert outcome.status == "expired"
    assert outcome.billable is True
    assert executor.calls == []


def test_expired_operation_honors_explicit_failure_billing_opt_out():
    executor = FakeExecutor(response(200, {"state": "COMPLETE"}))
    route_config = route({"poll": poll_config(billable_statuses=[])})
    expired = replace(lease(route_config), expires_at=NOW - timedelta(seconds=1))

    outcome = asyncio.run(make_poller(executor)._poll(expired))

    assert outcome.status == "expired"
    assert outcome.billable is False
    assert executor.calls == []


def test_expired_cancel_request_stops_retrying_and_settles_without_network():
    executor = FakeExecutor(response(429, {}))
    route_config = route(
        {
            "poll": poll_config(),
            "cancel": {
                "endpoint": {
                    "path_template": "/tasks/{task_id}",
                    "method": "DELETE",
                },
                "retry": {"statuses": [429], "max_attempts": 0},
            },
        }
    )
    expired = replace(
        lease(route_config),
        expires_at=NOW - timedelta(seconds=1),
        cancel_requested=True,
    )

    outcome = asyncio.run(make_poller(executor)._cancel_outcome(expired))

    assert outcome.status == "expired"
    assert outcome.billable is True
    assert executor.calls == []


def test_claim_statement_uses_skip_locked_and_short_due_filters():
    statement = build_claim_statement(NOW, 25)
    criteria = str(statement.whereclause).upper()

    assert statement._for_update_arg.skip_locked is True
    assert [table.name for table in statement._for_update_arg.of] == [
        "external_operations"
    ]
    assert "pending" in statement.compile().params.values()
    assert "created_at" in criteria.lower()
    assert "LEASE_EXPIRES_AT IS NULL" in criteria
    assert "NEXT_POLL_AT IS NULL" in criteria
    assert statement._limit_clause.value == 25


def test_claim_statement_only_recovers_non_task_sessions_after_hard_expiration():
    statement = build_claim_statement(NOW, 25)
    criteria = str(statement.whereclause).upper()
    parameters = statement.compile().params.values()

    assert any(
        isinstance(value, (list, tuple, set)) and {"stream", "realtime"}.issubset(value)
        for value in parameters
    )
    assert "EXPIRES_AT IS NOT NULL" in criteria
    assert "EXPIRES_AT <=" in criteria


def _running_session(
    *,
    operation_mode="stream",
    expires_at=NOW - timedelta(seconds=1),
    bill_partial_streams=True,
):
    route_config = route(
        {},
        operation_mode=operation_mode,
        response_config={"bill_partial_streams": bill_partial_streams},
    )
    return SimpleNamespace(
        operation_id="session-operation",
        user_id="session-user",
        account_id="session-account",
        operation_mode=operation_mode,
        status="running",
        expires_at=expires_at,
        usage={"requests": 1, "tokens": {"output": 7}},
        route_snapshot=route_config.model_dump(mode="json"),
        result_descriptor=None,
        poll_attempts=0,
        lease_owner=None,
        lease_expires_at=None,
        next_poll_at=expires_at,
        error=None,
        finished_at=None,
        last_polled_at=None,
    )


@pytest.mark.parametrize("expires_at", [None, NOW + timedelta(seconds=1)])
def test_session_recovery_never_reaps_an_unexpired_or_unarmed_session(expires_at):
    operation = _running_session(expires_at=expires_at)

    event = polling._terminalize_expired_session(operation, NOW)

    assert event is None
    assert operation.status == "running"


@pytest.mark.parametrize(
    ("operation_mode", "bill_partial_streams", "expected_billable"),
    [
        ("stream", True, True),
        ("stream", False, False),
        ("realtime", False, True),
    ],
)
def test_expired_session_recovery_uses_persisted_partial_work_policy(
    operation_mode, bill_partial_streams, expected_billable
):
    operation = _running_session(
        operation_mode=operation_mode,
        bill_partial_streams=bill_partial_streams,
    )

    event = polling._terminalize_expired_session(operation, NOW)

    assert event is not None
    assert event.billable is expected_billable
    assert event.usage.tokens == {"output": 7}
    assert event.error["code"] == "session_recovery_timeout"
    assert operation.status == "failed"
    assert operation.finished_at == NOW
    assert operation.next_poll_at is None
    assert operation.poll_attempts == 1


@pytest.mark.asyncio
async def test_usage_checkpoint_coalesces_observations_and_feeds_crash_reaper():
    operation = _running_session()
    current = NormalizedUsage(requests=1)
    writes = []
    persisted = asyncio.Event()

    async def persist_usage(value):
        writes.append(value)
        operation.usage = value
        persisted.set()

    checkpoints = UsageCheckpointLoop(
        operation_id=operation.operation_id,
        read_usage=lambda: current.to_dict(),
        persist_usage=persist_usage,
        initial_usage=current.to_dict(),
        interval_seconds=0.01,
    )
    checkpoints.start()
    # These updates model per-token/per-message observations. None of them writes
    # directly; the one periodic tick persists only the latest aggregate.
    for output_tokens in range(1, 51):
        current = NormalizedUsage(requests=1, tokens={"output": output_tokens})
    await asyncio.wait_for(persisted.wait(), timeout=1)
    await checkpoints.stop()

    assert len(writes) == 1
    assert writes[0]["tokens"] == {"output": "50"}

    event = polling._terminalize_expired_session(operation, NOW)

    assert event is not None
    assert event.usage.tokens == {"output": 50}


def test_claim_due_reaps_expired_session_without_task_transport_snapshot(monkeypatch):
    operation = _running_session()
    events = []
    governance_locks = []

    class Result:
        def unique(self):
            return self

        def scalars(self):
            return self

        def all(self):
            return [operation]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def begin(self):
            return self

        async def execute(self, _statement):
            return Result()

    async def settle(event):
        events.append(event)

    async def lock_scopes(_session, *, user_ids, account_ids):
        governance_locks.append((tuple(user_ids), tuple(account_ids)))

    monkeypatch.setattr(polling, "lock_governance_state_rows", lock_scopes)

    poller = ExternalOperationPoller(
        session_factory=Session,
        settlement_hook=settle,
        clock=lambda: NOW,
    )

    leases = asyncio.run(poller.claim_due())

    assert leases == ()
    assert [event.operation_id for event in events] == ["session-operation"]
    assert operation.status == "failed"
    assert governance_locks == [
        ((operation.user_id,), (operation.account_id,)),
    ]


def test_worker_settings_reject_unbounded_values():
    with pytest.raises(PollingConfigurationError):
        WorkerSettings(batch_size=0)
    with pytest.raises(PollingConfigurationError):
        WorkerSettings(lease_seconds=1)
    with pytest.raises(PollingConfigurationError):
        WorkerSettings(retention_batch_size=0)
    with pytest.raises(PollingConfigurationError):
        WorkerSettings(settlement_batch_size=0)


def test_retention_waits_for_every_advertised_artifact_expiration():
    descriptor = {
        "artifacts": [
            {"expires_at": (NOW - timedelta(seconds=1)).isoformat()},
            {"expires_at": (NOW + timedelta(seconds=1)).isoformat()},
        ]
    }

    assert polling._retained_artifacts_expired(descriptor, now=NOW) is False
    descriptor["artifacts"][1]["expires_at"] = NOW.isoformat()
    assert polling._retained_artifacts_expired(descriptor, now=NOW) is True


@pytest.mark.parametrize("expires_at", [None, "not-a-date", "2030-01-01T00:00:00"])
def test_retention_fails_closed_for_unbounded_artifact_expiration(expires_at):
    assert (
        polling._retained_artifacts_expired(
            {"artifacts": [{"expires_at": expires_at}]}, now=NOW
        )
        is False
    )


def test_governance_bucket_prune_is_bounded_and_lock_skipping():
    sql = str(
        build_governance_bucket_prune_statement(37).compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    ).upper()

    assert sql.startswith("WITH EXPIRED_EXTERNAL_GOVERNANCE_BUCKETS AS")
    assert "DELETE FROM EXTERNAL_GOVERNANCE_BUCKETS" in sql
    assert "FOR UPDATE OF EXTERNAL_GOVERNANCE_BUCKETS SKIP LOCKED" in sql
    assert "LIMIT 37" in sql
    assert "CLOCK_TIMESTAMP()" in sql
    assert "INTERVAL '24 HOURS 5 MINUTES'" in sql


@pytest.mark.asyncio
async def test_governance_bucket_maintenance_uses_grace_batch_and_metric(monkeypatch):
    statements = []
    metric_values = []

    class DeleteResult:
        rowcount = 4

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def begin(self):
            return Transaction()

        async def execute(self, statement):
            statements.append(statement)
            return DeleteResult()

    class Metric:
        def inc(self, value):
            metric_values.append(value)

    monkeypatch.setattr(polling, "governance_bucket_deletions", Metric())
    poller = ExternalOperationPoller(
        session_factory=Session,
        settings=WorkerSettings(retention_batch_size=23),
        clock=lambda: NOW,
    )

    deleted = await poller.collect_expired_governance_buckets()

    assert deleted == 4
    assert metric_values == [4]
    sql = str(
        statements[0].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    ).upper()
    assert "LIMIT 23" in sql
    assert "CLOCK_TIMESTAMP()" in sql


@pytest.mark.asyncio
async def test_governance_state_reconciliation_is_bounded_and_user_first():
    updates = []
    select_count = 0

    class Rows:
        def __init__(self, value):
            self.value = value

        def scalars(self):
            return self

        def all(self):
            return self.value if isinstance(self.value, list) else []

        def scalar_one_or_none(self):
            return self.value

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def begin(self):
            return Transaction()

        async def execute(self, statement, parameters=None):
            nonlocal select_count
            if statement.is_select:
                if "external_operations" in str(statement):
                    return Rows([])
                select_count += 1
                value = {
                    1: SimpleNamespace(scope_type="user", scope_id="user-id"),
                    2: None,
                    3: SimpleNamespace(scope_type="account", scope_id="account-id"),
                    4: None,
                }[select_count]
                return Rows(value)
            updates.append((str(statement), parameters))
            return Rows(None)

    poller = ExternalOperationPoller(
        session_factory=Session,
        settings=WorkerSettings(settlement_batch_size=7),
    )

    reconciled = await poller.reconcile_governance_state_once()

    assert reconciled == 2
    assert [parameters["scope_type"] for _, parameters in updates] == [
        "user",
        "account",
    ]
    assert all("balance_exempt" in sql for sql, _ in updates)


def test_missing_governance_scope_repair_is_bounded_to_unresolved_work():
    sql = str(
        build_missing_governance_scope_statement("user", 19).compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    ).upper()

    assert "FROM EXTERNAL_OPERATIONS" in sql
    assert "NOT (EXISTS" in sql
    assert "EXTERNAL_GOVERNANCE_STATE" in sql
    assert "SETTLEMENT_STATUS IN" in sql
    for status_name in ("'PENDING'", "'FAILED'", "'QUARANTINED'"):
        assert status_name in sql
    assert "LIMIT 19" in sql


@pytest.mark.asyncio
async def test_governance_reconciliation_recreates_a_missing_active_scope(monkeypatch):
    locked = []
    updates = []
    missing_queries = 0

    class Rows:
        def __init__(self, values):
            self.values = values

        def scalars(self):
            return self

        def all(self):
            return self.values

        def scalar_one_or_none(self):
            return None

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def begin(self):
            return Transaction()

        async def execute(self, statement, parameters=None):
            nonlocal missing_queries
            sql = str(statement)
            if statement.is_select and "external_operations" in sql:
                missing_queries += 1
                return Rows(["lost-user"] if missing_queries == 1 else [])
            if statement.is_select:
                return Rows([])
            updates.append((sql, parameters))
            return Rows([])

    async def lock_scopes(_db, *, user_ids, account_ids):
        locked.append((tuple(user_ids), tuple(account_ids)))

    monkeypatch.setattr(polling, "lock_governance_state_rows", lock_scopes)
    poller = ExternalOperationPoller(
        session_factory=Session,
        settings=WorkerSettings(settlement_batch_size=1),
    )

    reconciled = await poller.reconcile_governance_state_once()

    assert reconciled == 1
    assert locked == [(("lost-user",), ())]
    assert [parameters for _, parameters in updates] == [
        {"scope_type": "user", "scope_id": "lost-user"}
    ]


@pytest.mark.asyncio
async def test_retention_savepoint_isolates_one_fk_poisoned_row(monkeypatch):
    candidates = [
        SimpleNamespace(operation_id="poisoned", result_descriptor={"artifacts": []}),
        SimpleNamespace(
            operation_id="collectable", result_descriptor={"artifacts": []}
        ),
    ]
    delete_attempts = 0

    class SelectResult:
        def scalars(self):
            return candidates

    class DeleteResult:
        rowcount = 1

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def begin(self):
            return Transaction()

        def begin_nested(self):
            return Transaction()

        async def execute(self, statement):
            nonlocal delete_attempts
            if statement.is_select:
                return SelectResult()
            delete_attempts += 1
            if delete_attempts == 1:
                raise IntegrityError("DELETE", {}, RuntimeError("foreign key"))
            return DeleteResult()

    poller = ExternalOperationPoller(
        session_factory=Session,
        settings=WorkerSettings(retention_days=1, retention_batch_size=10),
        clock=lambda: NOW,
    )
    monkeypatch.setattr(polling, "lock_governance_state_rows", AsyncMock())

    deleted = await poller.collect_retained_operations()

    assert delete_attempts == 2
    assert deleted == 1


def test_start_and_stop_helpers_on_service_instance_are_idempotent():
    class IdlePoller(ExternalOperationPoller):
        async def poll_once(self):
            self._stop.set()
            return 0

    async def no_settlement(_event):
        return None

    async def exercise():
        poller = IdlePoller(settlement_hook=no_settlement)
        first = poller.start()
        second = poller.start()
        assert first is second
        await first
        await poller.stop()
        assert poller.running is False

    asyncio.run(exercise())
