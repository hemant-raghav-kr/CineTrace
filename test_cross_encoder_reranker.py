"""
CineTrace — Unit and Functional Test Suite for Cross-Encoder Reranking (Phase 5.7)
Validates offline mock inference, stable ID deduplication, head+tail passage selection,
score preservation, honest relevance labels, and clean fallback behavior.
"""

import unittest
from pathlib import Path
from typing import Any, Dict, List

import config
import semantic_service
from semantic_service import MockCrossEncoder, CrossEncoderService, select_passage_head_tail
import scene_matcher
from scene_matcher import SceneMatcher


class TestCrossEncoderReranker(unittest.TestCase):

    def test_passage_selection_short_plot(self):
        """Short plots under 350 words should be returned in full without truncation."""
        short_plot = "A small group of rebel fighters plan to steal the secret plans for the Death Star."
        passage = select_passage_head_tail(short_plot, max_head_words=220, max_tail_words=130)
        self.assertEqual(passage, short_plot)

    def test_passage_selection_long_plot_head_tail(self):
        """Long plots should preserve first 220 words (premise) and last 130 words (climax/resolution)."""
        words = [f"word_{i}" for i in range(500)]
        long_plot = " ".join(words)
        passage = select_passage_head_tail(long_plot, max_head_words=220, max_tail_words=130)
        
        self.assertTrue(passage.startswith("word_0 word_1"))
        self.assertTrue(passage.endswith("word_498 word_499"))
        self.assertIn("... [plot progression] ...", passage)
        passage_words = passage.split()
        # Head (220) + transition markers (4) + tail (130) = 354 words
        self.assertEqual(len(passage_words), 354)

    def test_mock_cross_encoder_deterministic_scoring(self):
        """Mock cross encoder produces deterministic, reproducible logits based on token overlap."""
        mock = MockCrossEncoder()
        query = "dream sharing inception"
        doc_relevant = "A thief steals corporate secrets through dream sharing technology."
        doc_irrelevant = "A romantic comedy set in Paris during the springtime."

        scores = mock.predict([(query, doc_relevant), (query, doc_irrelevant)])
        self.assertEqual(len(scores), 2)
        # Relevant document has overlap, irrelevant has zero overlap
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], 0.0)
        self.assertLess(scores[1], -2.0)

    def test_candidate_deduplication_by_id(self):
        """Duplicate candidates with identical catalogue IDs must be deduplicated before reranking."""
        ce_svc = CrossEncoderService()
        ce_svc.set_custom_model(MockCrossEncoder())

        candidates = [
            {"id": 101, "title": "Movie A", "plot": "Space journey", "similarity": 0.8},
            {"id": 102, "title": "Movie B", "plot": "Ocean voyage", "similarity": 0.75},
            {"id": 101, "title": "Movie A Duplicate", "plot": "Space journey duplicate", "similarity": 0.70},
            {"id": 103, "title": "Movie C", "plot": "Mountain climbing", "similarity": 0.65},
        ]

        reranked = ce_svc.rerank("space journey exploration", candidates, top_k=10)
        ids = [c["id"] for c in reranked]
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids.count(101), 1)

    def test_score_preservation_and_honest_labels(self):
        """Cross-encoder reranking preserves original first-stage scores and applies honest labels."""
        ce_svc = CrossEncoderService()
        ce_svc.set_custom_model(MockCrossEncoder())

        candidates = [
            {
                "id": 1,
                "title": "Dreamscape",
                "plot": "A psychic enters other people's dreams to help or hurt them.",
                "similarity": 0.55,
                "score_label": "Semantic Match: 55%",
                "match_type": "semantic"
            },
            {
                "id": 2,
                "title": "Inception",
                "plot": "A thief steals corporate secrets through dream sharing technology and inception.",
                "similarity": 0.45,
                "score_label": "Semantic Match: 45%",
                "match_type": "semantic"
            }
        ]

        reranked = ce_svc.rerank("dream sharing inception", candidates, top_k=2)

        # Inception should be boosted to #1 by cross-encoder
        self.assertEqual(reranked[0]["title"], "Inception")
        self.assertEqual(reranked[0]["original_rank"], 2)
        self.assertEqual(reranked[0]["original_similarity"], 0.45)
        self.assertEqual(reranked[0]["original_score_label"], "Semantic Match: 45%")
        self.assertTrue(reranked[0]["reranked"])
        self.assertEqual(reranked[0]["match_type"], "reranked")

        # Honest label check (no fake percentage or probability)
        score_label = reranked[0]["score_label"]
        self.assertTrue(score_label.startswith("Reranked Match: "))
        self.assertNotIn("%", score_label)
        self.assertNotIn("probability", score_label.lower())
        self.assertIn("Match:", score_label)

    def test_clean_fallback_when_disabled(self):
        """When rerank=False, SceneMatcher returns first-stage ranking unchanged."""
        records = [
            {"id": 1, "title": "The Matrix", "plot": "A computer hacker learns about the true nature of reality."},
            {"id": 2, "title": "Speed", "plot": "A young cop must prevent a bomb exploding on a city bus."}
        ]
        sm = SceneMatcher(records=records)
        results = sm.search("computer hacker matrix", rerank=False)
        self.assertTrue(len(results) > 0)
        self.assertNotEqual(results[0]["match_type"], "reranked")
        self.assertNotIn("raw_cross_encoder_score", results[0])

    def test_clean_fallback_when_model_unavailable(self):
        """When cross encoder model is unavailable or fails, SceneMatcher falls back cleanly."""
        records = [
            {"id": 1, "title": "The Matrix", "plot": "A computer hacker learns about the true nature of reality."},
            {"id": 2, "title": "Speed", "plot": "A young cop must prevent a bomb exploding on a city bus."}
        ]
        sm = SceneMatcher(records=records)
        # Create un-loaded service
        broken_svc = CrossEncoderService(model_name="non-existent-model-weights")
        results = sm.search("computer hacker matrix", rerank=True, cross_encoder_service=broken_svc)
        self.assertTrue(len(results) > 0)
        # Cleanly returned first stage results without raising exception
        self.assertNotEqual(results[0]["match_type"], "reranked")


if __name__ == "__main__":
    unittest.main()
