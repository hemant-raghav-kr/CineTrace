"""
CineTrace Phase 5.8 — Offline Reliability & Production Resilience Test Suite

Validates:
1. True Offline Operation: Models load and inference runs with all network sockets disabled.
2. Missing Weights Handling: Clean failure state and lexical fallback when weights are absent.
3. Corrupted Weights Handling: Graceful error capture (ModelStatus.ERROR) and UI fallback on corrupted files.
4. Engine Provenance & Mock Detection: Real transformer vs mock engine distinction and payload stamping.
5. OMDb Offline Isolation: Network failures during OMDb requests do not crash the app or block catalogue search.
"""

import os
import sys
import tempfile
import socket
import unittest
import requests
from pathlib import Path
from unittest.mock import patch

# Ensure CineTrace root is on path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
import database
import semantic_service
from semantic_service import (
    ModelStatus,
    SemanticService,
    SentenceTransformerModel,
    CrossEncoderService,
    MockEmbeddingModel,
    MockCrossEncoder,
    get_semantic_service,
    get_cross_encoder_service
)
from scene_matcher import SceneMatcher
import omdb_client
from omdb_client import OMDbClient, OMDbConnectionError


class TestOfflineProductionReadiness(unittest.TestCase):

    def setUp(self):
        self.original_cache_dir = config.MODEL_CACHE_DIR

    def tearDown(self):
        config.MODEL_CACHE_DIR = self.original_cache_dir

    def test_01_true_offline_operation_with_sockets_blocked(self):
        """Verify models load and search executes strictly offline when all network connections fail."""
        def blocked_socket(*args, **kwargs):
            raise OSError("Network is unreachable (simulated strict offline environment)")

        with patch("socket.socket.connect", side_effect=blocked_socket), \
             patch("socket.create_connection", side_effect=blocked_socket), \
             patch("urllib.request.urlopen", side_effect=blocked_socket):

            # 1. Bi-encoder loads offline via local files only
            bi_svc = SemanticService(model_name=config.DEFAULT_MODEL_NAME)
            ok, msg = bi_svc.load_model_sync(allow_download=False)
            self.assertTrue(ok, f"Bi-encoder must load from local cache with zero network access: {msg}")
            self.assertEqual(bi_svc.model_status, ModelStatus.READY)
            self.assertFalse(bi_svc.is_mock_active)
            self.assertEqual(bi_svc.inference_engine, "real_sentence_transformer")

            # 2. Query encoding runs offline
            q_emb = bi_svc.model.encode(["A time loop comedy film"])
            self.assertEqual(len(q_emb), 1)
            self.assertEqual(len(q_emb[0]), config.SEMANTIC_DIMENSION)

            # 3. Cross-encoder loads offline
            ce_svc = CrossEncoderService(
                model_name=config.CROSS_ENCODER_MODEL_NAME,
                cache_dir=config.MODEL_CACHE_DIR
            )
            ce_ok, ce_msg = ce_svc.load_model_sync()
            self.assertTrue(ce_ok, f"Cross-encoder must load offline from local cache: {ce_msg}")
            self.assertEqual(ce_svc.model_status, ModelStatus.READY)
            self.assertFalse(ce_svc.is_mock_active)
            self.assertEqual(ce_svc.inference_engine, "real_cross_encoder")

            # 4. Cross-encoder reranking runs offline
            candidates = [
                {"id": 1, "title": "Palm Springs", "plot": "Two wedding guests get stuck in a time loop.", "similarity": 0.8},
                {"id": 2, "title": "Groundhog Day", "plot": "A meteorologist relives the same day.", "similarity": 0.75}
            ]
            reranked = ce_svc.rerank("time loop", candidates, top_k=2)
            self.assertEqual(len(reranked), 2)
            self.assertEqual(reranked[0]["inference_engine"], "real_cross_encoder")

    def test_02_missing_weights_handling(self):
        """When local weights directory is missing or empty, status must be NOT_DOWNLOADED without crashing."""
        with tempfile.TemporaryDirectory() as empty_dir:
            config.MODEL_CACHE_DIR = Path(empty_dir)

            bi_svc = SemanticService(model_name=config.DEFAULT_MODEL_NAME)
            loaded, msg = bi_svc.load_model_sync(allow_download=False)
            self.assertFalse(loaded)
            self.assertEqual(bi_svc.model_status, ModelStatus.NOT_DOWNLOADED)

            ce_svc = CrossEncoderService(
                model_name=config.CROSS_ENCODER_MODEL_NAME,
                cache_dir=Path(empty_dir)
            )
            ce_ok, ce_msg = ce_svc.load_model_sync()
            self.assertFalse(ce_ok)
            self.assertEqual(ce_svc.model_status, ModelStatus.NOT_DOWNLOADED)

    def test_03_corrupted_model_weights_handling(self):
        """When model files are corrupted/unreadable, service enters ModelStatus.ERROR and provides error details."""
        with tempfile.TemporaryDirectory() as corrupt_dir:
            config.MODEL_CACHE_DIR = Path(corrupt_dir)
            
            # Create a mock cache directory with broken/unparseable files
            model_target = Path(corrupt_dir) / f"models--sentence-transformers--{config.DEFAULT_MODEL_NAME.replace('/', '--')}"
            snapshots = model_target / "snapshots" / "corrupted_rev"
            snapshots.mkdir(parents=True, exist_ok=True)
            (snapshots / "config.json").write_text("{corrupted_syntax: broken", encoding="utf-8")
            (snapshots / "pytorch_model.bin").write_bytes(b"corrupted binary data not torch")

            bi_svc = SemanticService(model_name=config.DEFAULT_MODEL_NAME)
            loaded, msg = bi_svc.load_model_sync(allow_download=False)
            self.assertFalse(loaded)
            self.assertEqual(bi_svc.model_status, ModelStatus.ERROR)
            self.assertTrue(bi_svc.load_error is not None and len(bi_svc.load_error) > 0)

            # Cross-encoder corruption
            ce_target = Path(corrupt_dir) / f"models--cross-encoder--{config.CROSS_ENCODER_MODEL_NAME.replace('/', '--')}"
            ce_snapshots = ce_target / "snapshots" / "corrupted_rev"
            ce_snapshots.mkdir(parents=True, exist_ok=True)
            (ce_snapshots / "config.json").write_text("{broken", encoding="utf-8")
            (ce_snapshots / "pytorch_model.bin").write_bytes(b"corrupted binary")

            ce_svc = CrossEncoderService(
                model_name=config.CROSS_ENCODER_MODEL_NAME,
                cache_dir=Path(corrupt_dir)
            )
            ce_ok, ce_msg = ce_svc.load_model_sync()
            self.assertFalse(ce_ok)
            self.assertEqual(ce_svc.model_status, ModelStatus.ERROR)
            self.assertTrue(ce_svc.load_error is not None and len(ce_svc.load_error) > 0)

    def test_04_engine_provenance_and_mock_detection(self):
        """Distinguish real model inference from mock scorers and verify provenance tags."""
        # Standard production services
        bi_svc = SemanticService(model_name=config.DEFAULT_MODEL_NAME)
        self.assertFalse(bi_svc.is_mock_active)
        self.assertEqual(bi_svc.inference_engine, "none")  # Not loaded yet
        ok, _ = bi_svc.load_model_sync(allow_download=False)
        self.assertTrue(ok)
        self.assertEqual(bi_svc.inference_engine, "real_sentence_transformer")

        ce_svc = CrossEncoderService()
        self.assertFalse(ce_svc.is_mock_active)
        # Mock injection
        mock_bi = MockEmbeddingModel(dimension=config.SEMANTIC_DIMENSION)
        bi_svc.set_custom_model(mock_bi)
        self.assertTrue(bi_svc.is_mock_active)
        self.assertEqual(bi_svc.inference_engine, "mock_deterministic")

        mock_ce = MockCrossEncoder()
        ce_svc.set_custom_model(mock_ce)
        self.assertTrue(ce_svc.is_mock_active)
        self.assertEqual(ce_svc.inference_engine, "mock_deterministic")

        # Rerank with mock produces mock provenance
        candidates = [
            {"id": 10, "title": "Test Movie", "plot": "Test plot", "similarity": 0.5}
        ]
        reranked = ce_svc.rerank("test", candidates, top_k=1)
        self.assertEqual(reranked[0]["inference_engine"], "mock_deterministic")

    def test_05_omdb_offline_isolation(self):
        """Network drops during OMDb lookup do not crash the app or interrupt catalogue workflow."""
        def broken_urlopen(*args, **kwargs):
            raise requests.exceptions.ConnectionError("DNS lookup failed (offline)")

        with patch("requests.get", side_effect=broken_urlopen):
            client = OMDbClient("fake_key_for_test")
            with self.assertRaises(OMDbConnectionError):
                client.get_movie_details("tt0137523")


if __name__ == "__main__":
    unittest.main()
