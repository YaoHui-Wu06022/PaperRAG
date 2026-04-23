from __future__ import annotations

import urllib.error
import urllib.request
import unittest
from unittest.mock import patch

from paper_rag.dataprocess.metadata.retry import urlopen_with_retry


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


class MetadataRetryTests(unittest.TestCase):
    def test_retries_timeout_once(self) -> None:
        request = urllib.request.Request("https://example.test")
        response = Response()
        with patch(
            "paper_rag.dataprocess.metadata.retry.urllib.request.urlopen",
            side_effect=[TimeoutError("timed out"), response],
        ) as urlopen:
            self.assertIs(urlopen_with_retry(request, timeout=1, delay_seconds=0), response)
            self.assertEqual(urlopen.call_count, 2)

    def test_retries_http_429_once(self) -> None:
        request = urllib.request.Request("https://example.test")
        response = Response()
        error = urllib.error.HTTPError("https://example.test", 429, "Too Many Requests", {}, None)
        with patch(
            "paper_rag.dataprocess.metadata.retry.urllib.request.urlopen",
            side_effect=[error, response],
        ) as urlopen:
            self.assertIs(urlopen_with_retry(request, timeout=1, delay_seconds=0), response)
            self.assertEqual(urlopen.call_count, 2)

    def test_does_not_retry_non_retriable_http_error(self) -> None:
        request = urllib.request.Request("https://example.test")
        error = urllib.error.HTTPError("https://example.test", 404, "Not Found", {}, None)
        with patch(
            "paper_rag.dataprocess.metadata.retry.urllib.request.urlopen",
            side_effect=error,
        ) as urlopen:
            with self.assertRaises(urllib.error.HTTPError):
                urlopen_with_retry(request, timeout=1, delay_seconds=0)
            self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
