"""
CineTrace Phase 5.6.2 - GUI Semantic Lifecycle Test Suite

Validates:
1. Semantic-model cold start and loading indicator
2. Delayed loading resolution without main-thread blocking
3. Warm start immediate search execution
4. Loading failure with actionable fallback card (Lexical Fallback + Manage Semantic Search)
5. Rapid successive queries with generation counter race protection
6. Displayed percentage verified as similarity/relevance score, not probability
"""

import sys
import time
import tkinter as tk
from unittest.mock import MagicMock, patch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from main import CineTraceApp
import main


def wait_for_search(app, expected_gen, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.update()
        if not app.scene_is_searching and app.scene_search_generation == expected_gen and app.scene_has_searched:
            return True
        time.sleep(0.05)
    return False


def run_gui_lifecycle_tests():
    print("=" * 80)
    print("RUNNING CINETRACE GUI SEMANTIC LIFECYCLE TEST SUITE")
    print("=" * 80)

    app = CineTraceApp()
    app.withdraw()  # Off-screen for headless verification
    app.update()

    try:
        # Wait for background initialization of SceneMatcher to complete cleanly
        print("Waiting for catalogue engine initialization...")
        while app.scene_dataset_status == "Loading dataset...":
            app.update()
            time.sleep(0.05)
        print(f"Catalogue Engine Ready: {app.scene_dataset_status}")

        # Switch to Scene Finder tab
        print("\n--- Test 1: Navigation to Scene Finder ---")
        app._switch_tab("Scene Finder")
        app.update()
        time.sleep(0.3)
        app.update()
        assert app.active_tab == "Scene Finder"
        print("[PASS] Switched to Scene Finder tab successfully.")

        # Test 2: Cold start & delayed loading
        print("\n--- Test 2: Cold Start & Delayed Loading UI Coordination ---")
        mock_sem = MagicMock()
        sem_state = {"ready": False}
        mock_sem.is_ready.side_effect = lambda: sem_state["ready"]
        mock_sem.is_loading_model.return_value = True
        mock_sem.is_model_loaded.return_value = False
        mock_sem.is_available.return_value = True

        def _mock_wait(timeout=15.0):
            time.sleep(0.2)
            sem_state["ready"] = True
            return True
        mock_sem.wait_until_ready.side_effect = _mock_wait
        mock_sem.search_by_query.return_value = {57: 0.5714}

        mock_local_info = {"found": True, "path": "models/all-MiniLM-L6-v2", "size_mb": 80.0}

        with patch.object(main.semantic_service, "get_semantic_service", return_value=mock_sem), \
             patch.object(main.semantic_service, "get_local_model_info", return_value=mock_local_info):

            app.scene_matcher.semantic_service = mock_sem
            app.scene_results = []
            app.scene_text_box.delete("1.0", "end")
            app.scene_text_box.insert("1.0", "A young girl with extraordinary mathematical abilities")
            app.scene_search_mode_var.set("Scene / Plot (Semantic)")

            # Trigger search
            app._on_scene_search_clicked()
            app.update()

            # Immediate UI verification on main thread
            status_text = app.scene_status_lbl.cget("text")
            btn_text = app.scene_search_btn.cget("text")
            btn_state = str(app.scene_search_btn.cget("state"))

            assert "Loading semantic model" in status_text, f"Expected loading message, got: {status_text}"
            assert btn_text == "Loading Model...", f"Expected 'Loading Model...', got: {btn_text}"
            assert btn_state == "disabled", f"Expected disabled button, got: {btn_state}"
            print(f"  -> Immediate UI State: [{btn_text} | state={btn_state}]")
            print(f"  -> Status Message: '{status_text}'")

            # Allow worker to finish waiting and execute search
            gen2 = app.scene_search_generation
            assert wait_for_search(app, gen2), "Search did not complete in time"

            assert app.scene_search_btn.cget("text") == "Find Movie"
            assert str(app.scene_search_btn.cget("state")) == "normal"
            assert len(app.scene_results) > 0, "Expected search results"
            print(f"  -> Top result returned: {[r['title'] for r in app.scene_results[:3]]}")
            assert app.scene_results[0]["title"] == "Gifted", f"Expected Gifted, got {app.scene_results[0]['title']}"
            print(f"  -> Post-Load Top Match: '{app.scene_results[0]['title']}' ({app.scene_results[0]['score_label']})")
            print("[PASS] Cold start delayed loading displayed responsive indicator and auto-executed on ready.")

        # Test 3: Warm start immediate execution
        print("\n--- Test 3: Warm Start Immediate Execution ---")
        mock_sem_warm = MagicMock()
        mock_sem_warm.is_ready.return_value = True
        mock_sem_warm.is_loading_model.return_value = False
        mock_sem_warm.is_available.return_value = True
        mock_sem_warm.search_by_query.return_value = {3: 0.53}

        with patch.object(main.semantic_service, "get_semantic_service", return_value=mock_sem_warm):
            app.scene_matcher.semantic_service = mock_sem_warm
            app.scene_results = []
            app.scene_text_box.delete("1.0", "end")
            app.scene_text_box.insert("1.0", "Wedding guests stuck in a time loop in the desert")
            app.scene_search_mode_var.set("Scene / Plot (Semantic)")

            app._on_scene_search_clicked()
            app.update()

            status_warm = app.scene_status_lbl.cget("text")
            assert "Loading semantic model" not in status_warm, "Warm start should not show loading message"
            assert app.scene_search_btn.cget("text") == "Searching..."

            gen3 = app.scene_search_generation
            assert wait_for_search(app, gen3), "Search did not complete in time"

            assert len(app.scene_results) > 0
            assert app.scene_results[0]["title"] == "Palm Springs"
            print(f"  -> Warm Start Top Match: '{app.scene_results[0]['title']}' ({app.scene_results[0]['score_label']})")
            print("[PASS] Warm start executed immediately without delay message.")

        # Test 4: Loading failure with actionable fallback UI
        print("\n--- Test 4: Loading Failure & Actionable Fallback Rendering ---")
        mock_sem_fail = MagicMock()
        mock_sem_fail.is_ready.return_value = False
        mock_sem_fail.is_loading_model.return_value = False
        mock_sem_fail.is_available.return_value = True
        mock_sem_fail.wait_until_ready.return_value = False

        mock_local_missing = {"found": False, "path": "", "size_mb": 0.0}

        with patch.object(main.semantic_service, "get_semantic_service", return_value=mock_sem_fail), \
             patch.object(main.semantic_service, "get_local_model_info", return_value=mock_local_missing):

            app.scene_matcher.semantic_service = mock_sem_fail
            app.scene_results = []
            app.scene_text_box.delete("1.0", "end")
            app.scene_text_box.insert("1.0", "Astronaut stranded on Mars surviving with botany")
            app.scene_search_mode_var.set("Scene / Plot (Semantic)")

            app._on_scene_search_clicked()
            app.update()

            def extract_widgets(parent):
                labels, buttons = [], []
                for child in parent.winfo_children():
                    if isinstance(child, tk.Label):
                        labels.append(child.cget("text"))
                    elif isinstance(child, tk.Button):
                        buttons.append(child.cget("text"))
                    sub_l, sub_b = extract_widgets(child)
                    labels.extend(sub_l)
                    buttons.extend(sub_b)
                return labels, buttons

            labels, buttons = extract_widgets(app.scene_results_frame)
            assert any("Semantic Search Engine Unavailable" in l for l in labels), f"No failure banner found in {labels}"
            assert any("⚡ Search with Lexical Fallback" in b for b in buttons), f"No lexical fallback btn in {buttons}"
            assert any("⚙️ Manage Semantic Search" in b for b in buttons), f"No manage btn in {buttons}"
            print("  -> Actionable Card Elements Found:")
            print("     • Banner: '⚠️ Semantic Search Engine Unavailable'")
            print("     • Button: '⚡ Search with Lexical Fallback'")
            print("     • Button: '⚙️ Manage Semantic Search'")

            # Execute lexical fallback action
            app.scene_results = []
            app._execute_lexical_fallback("Astronaut stranded on Mars", "All", None, None, 0.05)
            gen_fallback = app.scene_search_generation
            assert wait_for_search(app, gen_fallback), "Fallback search did not complete in time"

            assert app.scene_has_searched
            assert app.scene_search_btn.cget("text") == "Find Movie"
            print(f"  -> Lexical Fallback executed cleanly; returned {len(app.scene_results)} candidates.")
            print("[PASS] Loading failure renders actionable fallback card and allows immediate lexical search.")

        # Test 5: Rapid successive searches & race protection
        print("\n--- Test 5: Rapid Successive Searches & Generation Race Guard ---")
        start_gen = app.scene_search_generation

        app.scene_text_box.delete("1.0", "end")
        app.scene_text_box.insert("1.0", "Rapid Query 1")
        app._on_scene_search_clicked()

        app.scene_text_box.delete("1.0", "end")
        app.scene_text_box.insert("1.0", "Rapid Query 2")
        app._on_scene_search_clicked()

        app.scene_text_box.delete("1.0", "end")
        app.scene_text_box.insert("1.0", "Rapid Query 3")
        app._on_scene_search_clicked()

        assert app.scene_search_generation == start_gen + 3, f"Generation mismatch: {app.scene_search_generation}"

        # Inject late-arriving stale results from Generation (start_gen + 1)
        stale_results = [{"title": "Phantom Stale Movie", "similarity": 0.99, "score_label": "Semantic Match: 99%"}]
        app._on_scene_results_ready(stale_results, start_gen + 1)
        app.update()

        assert app.scene_results != stale_results, "Stale results from earlier generation should be discarded!"
        print(f"  -> Rapid generation advance: {start_gen} -> {app.scene_search_generation}")
        print(f"  -> Stale callback from generation {start_gen + 1} safely rejected.")
        print("[PASS] Generation counters strictly protect against race conditions and stale overwrites.")

        # Test 6: Score label verification (Similarity/Relevance, Not Probability)
        print("\n--- Test 6: Label Verification (Similarity/Relevance, Not Probability) ---")
        mock_sem_labels = MagicMock()
        mock_sem_labels.is_ready.return_value = True
        mock_sem_labels.is_loading_model.return_value = False
        mock_sem_labels.is_available.return_value = True
        mock_sem_labels.search_by_query.return_value = {57: 0.5714}

        with patch.object(main.semantic_service, "get_semantic_service", return_value=mock_sem_labels):
            app.scene_matcher.semantic_service = mock_sem_labels
            app.scene_results = []
            app.scene_text_box.delete("1.0", "end")
            app.scene_text_box.insert("1.0", "mathematical prodigy niece raised by uncle")
            app.scene_search_mode_var.set("Scene / Plot (Semantic)")

            app._on_scene_search_clicked()
            gen_labels = app.scene_search_generation
            assert wait_for_search(app, gen_labels), "Label search did not complete in time"

            all_labels, _ = extract_widgets(app.scene_results_frame)
            
            # Verify similarity header
            has_sim_header = any("text similarity score" in l.lower() for l in all_labels)
            assert has_sim_header, f"Header should state text similarity score in {all_labels}"

            # Verify no probability terminology
            for l in all_labels:
                assert "probability" not in l.lower(), f"Forbidden word 'probability' found in label: '{l}'"
                assert "% chance" not in l.lower(), f"Forbidden phrase '% chance' found in label: '{l}'"

            # Check candidate card labels
            card_score_labels = [r["score_label"] for r in app.scene_results if "score_label" in r]
            print(f"  -> Rendered candidate score labels: {card_score_labels[:3]}")
            for sl in card_score_labels:
                assert "Match:" in sl or "Plot" in sl or "Semantic" in sl or "Title" in sl or "Cast" in sl
                assert "probability" not in sl.lower()
            print("  -> Section Header Verified: 'Ranked by text similarity score'")
            print("  -> Percentage Verified as: 'Semantic Match: XX%' / 'Plot Match: XX%' (0 probability labels)")
            print("[PASS] All displayed scores are accurately labelled as text similarity/relevance scores.")

        print("\n" + "=" * 80)
        print("ALL 6 GUI SEMANTIC LIFECYCLE TESTS PASSED SUCCESSFULLY!")
        print("=" * 80)

    finally:
        app.destroy()


if __name__ == "__main__":
    run_gui_lifecycle_tests()
