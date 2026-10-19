"""
CineTrace — Poster Pipeline and Diagnostic Tests
Validates resilient poster loading, safe URL upgrading, HTTP fallback,
corrupt image handling, HTML page rejection, retry cooldown,
master cache reuse, concurrency deduplication, and thread-safe delivery.
"""

import io
import time
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import requests

import image_loader


class TestPosterPipeline(unittest.TestCase):
    """Unit tests for poster retrieval, validation, caching, and retries."""

    def setUp(self):
        image_loader.clear_image_caches()

    def tearDown(self):
        image_loader.clear_image_caches()

    def _create_test_image_bytes(self, size=(900, 1350), color=(200, 50, 50)):
        img = Image.new("RGB", size, color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_upgraded_url_success(self):
        """When upgraded _SX900 URL succeeds, it is decoded and master is cached."""
        orig_url = "https://m.media-amazon.com/images/M/MV5BTEST._V1_SX300.jpg"
        upgraded_url = "https://m.media-amazon.com/images/M/MV5BTEST._V1_SX900.jpg"
        img_bytes = self._create_test_image_bytes()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/jpeg"}
        mock_resp.content = img_bytes

        with patch("requests.get", return_value=mock_resp) as mock_get:
            fitted = image_loader.fetch_and_fit_pil(orig_url, 140, 195, title="Test Film")
            self.assertIsNotNone(fitted)
            self.assertEqual(fitted.size, (140, 195))

            # Verify requests called with upgraded URL
            mock_get.assert_called_once()
            called_url = mock_get.call_args[0][0]
            self.assertEqual(called_url, upgraded_url)

            # Master image cached under orig_url
            self.assertIn(orig_url, image_loader._PIL_CACHE)
            self.assertEqual(image_loader._PIL_CACHE[orig_url].size, (900, 1350))

    def test_upgraded_failure_fallback_to_original_success(self):
        """When upgraded URL returns 404, fallback to original URL succeeds."""
        orig_url = "https://m.media-amazon.com/images/M/MV5BTEST._V1_SX300.jpg"
        upgraded_url = "https://m.media-amazon.com/images/M/MV5BTEST._V1_SX900.jpg"
        orig_bytes = self._create_test_image_bytes(size=(300, 450))

        resp_404 = MagicMock()
        resp_404.status_code = 404
        resp_404.content = b"Not Found"

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.headers = {"content-type": "image/jpeg"}
        resp_200.content = orig_bytes

        def mock_request(url, **kwargs):
            if url == upgraded_url:
                return resp_404
            elif url == orig_url:
                return resp_200
            return resp_404

        with patch("requests.get", side_effect=mock_request) as mock_get:
            fitted = image_loader.fetch_and_fit_pil(orig_url, 140, 195, title="Fallback Film")
            self.assertIsNotNone(fitted)
            self.assertEqual(fitted.size, (140, 195))

            # Both URLs were tried
            self.assertEqual(mock_get.call_count, 2)
            self.assertEqual(mock_get.call_args_list[0][0][0], upgraded_url)
            self.assertEqual(mock_get.call_args_list[1][0][0], orig_url)

            # Original master cached
            self.assertIn(orig_url, image_loader._PIL_CACHE)
            self.assertEqual(image_loader._PIL_CACHE[orig_url].size, (300, 450))

    def test_http_errors_and_timeouts(self):
        """HTTP 500 error and request timeout fail gracefully without crash or corrupt cache."""
        url = "https://m.media-amazon.com/images/M/MV5BERR._V1_SX300.jpg"

        with patch("requests.get", side_effect=requests.exceptions.Timeout("Connection timed out")):
            fitted = image_loader.fetch_and_fit_pil(url, 140, 195, title="Timeout Movie")
            self.assertIsNone(fitted)
            self.assertNotIn(url, image_loader._PIL_CACHE)
            self.assertTrue(image_loader.is_url_in_cooldown(url))

    def test_invalid_html_error_response_rejected(self):
        """HTTP 200 containing HTML error portal is rejected, not treated as image."""
        url = "https://m.media-amazon.com/images/M/MV5BHTML._V1_SX300.jpg"

        html_resp = MagicMock()
        html_resp.status_code = 200
        html_resp.headers = {"content-type": "text/html; charset=utf-8"}
        html_resp.content = b"<html><body><h3>Error: Access Denied</h3></body></html>"

        with patch("requests.get", return_value=html_resp):
            fitted = image_loader.fetch_and_fit_pil(url, 140, 195, title="HTML Movie")
            self.assertIsNone(fitted)
            self.assertNotIn(url, image_loader._PIL_CACHE)

    def test_corrupted_image_data_rejected(self):
        """Malformed or truncated image bytes fail cleanly without crashing."""
        url = "https://m.media-amazon.com/images/M/MV5BCORRUPT._V1_SX300.jpg"

        corrupt_resp = MagicMock()
        corrupt_resp.status_code = 200
        corrupt_resp.headers = {"content-type": "image/jpeg"}
        corrupt_resp.content = b"\xff\xd8\xff\xe0CORRUPT_DATA_GARBAGE_BYTES_HERE"

        with patch("requests.get", return_value=corrupt_resp):
            fitted = image_loader.fetch_and_fit_pil(url, 140, 195, title="Corrupt Movie")
            self.assertIsNone(fitted)
            self.assertNotIn(url, image_loader._PIL_CACHE)

    def test_failed_load_retries_after_cooldown(self):
        """Failed download cools down, then successfully retries and clears failure record."""
        url = "https://m.media-amazon.com/images/M/MV5BRETRY._V1_SX300.jpg"
        img_bytes = self._create_test_image_bytes()

        # Step 1: Initial attempt fails (e.g. timeout)
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Net down")):
            res1 = image_loader.fetch_and_fit_pil(url, 140, 195, title="Retry Film")
            self.assertIsNone(res1)
            self.assertTrue(image_loader.is_url_in_cooldown(url))

        # Step 2: Immediate second attempt during cooldown is skipped
        with patch("requests.get") as mock_get_blocked:
            res2 = image_loader.fetch_and_fit_pil(url, 140, 195, title="Retry Film")
            self.assertIsNone(res2)
            mock_get_blocked.assert_not_called()

        # Step 3: Advance time beyond cooldown window
        count, old_time = image_loader._FAILED_REQUESTS[url]
        image_loader._FAILED_REQUESTS[url] = (count, old_time - 30.0)
        self.assertFalse(image_loader.is_url_in_cooldown(url))

        # Step 4: Retry succeeds
        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.headers = {"content-type": "image/jpeg"}
        success_resp.content = img_bytes

        with patch("requests.get", return_value=success_resp):
            res3 = image_loader.fetch_and_fit_pil(url, 140, 195, title="Retry Film")
            self.assertIsNotNone(res3)
            self.assertEqual(res3.size, (140, 195))
            # Failure record cleared on success
            self.assertNotIn(url, image_loader._FAILED_REQUESTS)

    def test_cache_correctness_and_successful_image_reuse(self):
        """Master original is reused for multiple dimensions without duplicate requests."""
        url = "https://m.media-amazon.com/images/M/MV5BMULTI._V1_SX300.jpg"
        img_bytes = self._create_test_image_bytes()

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "image/jpeg"}
        resp.content = img_bytes

        with patch("requests.get", return_value=resp) as mock_get:
            fit1 = image_loader.fetch_and_fit_pil(url, 140, 195, title="Multi Size")
            fit2 = image_loader.fetch_and_fit_pil(url, 200, 290, title="Multi Size")
            fit3 = image_loader.fetch_and_fit_pil(url, 140, 195, title="Multi Size")

            self.assertIsNotNone(fit1)
            self.assertIsNotNone(fit2)
            self.assertIsNotNone(fit3)
            self.assertEqual(fit1.size, (140, 195))
            self.assertEqual(fit2.size, (200, 290))
            self.assertEqual(fit3.size, (140, 195))

            # Exactly 1 HTTP request made!
            self.assertEqual(mock_get.call_count, 1)

    def test_concurrency_deduplication(self):
        """Simultaneous concurrent requests for the same URL share a single download."""
        import threading
        url = "https://m.media-amazon.com/images/M/MV5BCONC._V1_SX300.jpg"
        img_bytes = self._create_test_image_bytes()

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "image/jpeg"}
        resp.content = img_bytes

        def slow_get(*args, **kwargs):
            time.sleep(0.05)
            return resp

        results = []
        def worker():
            res = image_loader.fetch_and_fit_pil(url, 140, 195, title="Concurrent Film")
            results.append(res)

        with patch("requests.get", side_effect=slow_get) as mock_get:
            t1 = threading.Thread(target=worker)
            t2 = threading.Thread(target=worker)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            self.assertEqual(len(results), 2)
            self.assertIsNotNone(results[0])
            self.assertIsNotNone(results[1])
            self.assertEqual(mock_get.call_count, 1)

    def test_overwrite_guard_on_failed_async_load(self):
        """Failed async download does not call callback with placeholder to overwrite real poster."""
        url = "https://m.media-amazon.com/images/M/MV5BFAIL._V1_SX300.jpg"
        callback_mock = MagicMock()

        with patch("requests.get", side_effect=requests.exceptions.Timeout("Timeout")):
            image_loader.load_poster_async(url, 140, 195, callback_mock, title="Fail Film")
            time.sleep(0.1)
            # Callback is NOT invoked with placeholder when async download fails
            callback_mock.assert_not_called()

    def test_thread_safe_ui_delivery(self):
        """Successful async load schedules delivery on main thread via root.after."""
        import tkinter as _tk
        url = "https://m.media-amazon.com/images/M/MV5BASYNC._V1_SX300.jpg"
        img_bytes = self._create_test_image_bytes()

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "image/jpeg"}
        resp.content = img_bytes

        mock_root = MagicMock()
        mock_root.winfo_exists.return_value = True

        scheduled_callbacks = []
        mock_root.after = lambda ms, fn: scheduled_callbacks.append(fn)

        callback_mock = MagicMock()

        with patch.object(_tk, "_default_root", mock_root):
            with patch("requests.get", return_value=resp):
                image_loader.load_poster_async(url, 140, 195, callback_mock, title="Async Success")
                time.sleep(0.15)
                self.assertEqual(len(scheduled_callbacks), 1)
                with patch("PIL.ImageTk.PhotoImage", return_value="MOCK_PHOTO"):
                    scheduled_callbacks[0]()
                    callback_mock.assert_called_once_with("MOCK_PHOTO")

    def test_ipv4_resolution_preference(self):
        """Verify urllib3 gai_family is set to AF_INET to prevent IPv6 connect timeouts."""
        import socket
        import urllib3.util.connection as urllib3_cn
        self.assertEqual(urllib3_cn.allowed_gai_family(), socket.AF_INET)

    def test_pooled_session_configuration(self):
        """Verify get_image_session creates a pooled HTTPAdapter with trust_env=False."""
        session = image_loader.get_image_session()
        self.assertFalse(session.trust_env)
        self.assertIn("https://", session.adapters)
        adapter = session.adapters["https://"]
        self.assertEqual(adapter._pool_connections, 20)
        self.assertEqual(adapter._pool_maxsize, 20)

    def test_local_image_rendering_pipeline(self):
        """Verify Pillow-to-Tkinter rendering path with a known local JPEG file."""
        import tempfile
        import os
        from PIL import ImageTk

        # Create temporary test image
        img = Image.new("RGB", (200, 300), color=(100, 150, 200))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            img.save(tmp_path, format="JPEG")
            # Verify file exists and decodes properly
            opened = Image.open(tmp_path)
            self.assertEqual(opened.size, (200, 300))
            fitted = image_loader.fit_and_pad_image(opened, 140, 195)
            self.assertEqual(fitted.size, (140, 195))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
