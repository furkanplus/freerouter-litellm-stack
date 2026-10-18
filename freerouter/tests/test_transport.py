from __future__ import annotations

import http.client
import urllib.request
from typing import TYPE_CHECKING

from freerouter.transport import request

if TYPE_CHECKING:
    import pytest


def _raise(error: Exception):  # noqa: ANN202
    def opener(*_args: object, **_kwargs: object) -> None:
        raise error

    return opener


def test_a_dropped_connection_becomes_a_status_zero_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(urllib.request, "urlopen", _raise(http.client.IncompleteRead(b"")))

    # When
    response = request("https://example.com/v1/models")

    # Then
    assert response.status == 0


def test_a_garbled_status_line_becomes_a_status_zero_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(urllib.request, "urlopen", _raise(http.client.BadStatusLine("x")))

    # When
    response = request("https://example.com/v1/models")

    # Then
    assert response.status == 0
