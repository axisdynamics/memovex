"""Tests for WisdomStore — curation pipeline."""

import pytest
from memovex.core.wisdom_store import WisdomStore, WisdomLevel


class TestWisdomStoreRegistration:
    def test_register_returns_entry(self):
        ws = WisdomStore()
        entry = ws.register("m1", confidence=0.3, salience=0.3)
        assert entry.memory_id == "m1"
        assert entry.level == WisdomLevel.RAW

    def test_register_auto_promotes_to_processed(self):
        ws = WisdomStore()
        entry = ws.register("m1", confidence=0.45, salience=0.3)
        assert entry.level == WisdomLevel.PROCESSED

    def test_register_auto_promotes_to_curated(self):
        ws = WisdomStore()
        entry = ws.register("m1", confidence=0.65, salience=0.3)
        ws.corroborate("m1")
        assert ws.get_level("m1") == WisdomLevel.CURATED

    def test_register_auto_promotes_to_wisdom(self):
        ws = WisdomStore()
        ws.register("m1", confidence=0.85, salience=0.75)
        ws.corroborate("m1")
        ws.corroborate("m1")
        assert ws.is_wisdom("m1")


class TestWisdomStoreCorroboration:
    def test_corroborate_increments_evidence(self):
        ws = WisdomStore()
        ws.register("m1", confidence=0.5)
        ws.corroborate("m1")
        assert ws._entries["m1"].evidence_count == 1

    def test_corroborate_increases_confidence(self):
        ws = WisdomStore()
        ws.register("m1", confidence=0.5)
        ws.corroborate("m1", delta_confidence=0.1)
        assert ws._entries["m1"].confidence == pytest.approx(0.6)

    def test_corroborate_confidence_capped_at_one(self):
        ws = WisdomStore()
        ws.register("m1", confidence=0.98)
        ws.corroborate("m1", delta_confidence=0.1)
        assert ws._entries["m1"].confidence <= 1.0

    def test_corroborate_unknown_id_no_error(self):
        ws = WisdomStore()
        ws.corroborate("nonexistent")  # should not raise


class TestWisdomStoreScoring:
    def test_wisdom_score_raw(self):
        ws = WisdomStore()
        ws.register("m1", confidence=0.1)
        assert ws.wisdom_score("m1") == 0.0

    def test_wisdom_score_processed(self):
        ws = WisdomStore()
        ws.register("m1", confidence=0.45)
        assert ws.wisdom_score("m1") == pytest.approx(0.3)

    def test_wisdom_score_curated(self):
        ws = WisdomStore()
        ws.register("m1", confidence=0.65)
        ws.corroborate("m1")
        assert ws.wisdom_score("m1") == pytest.approx(0.6)

    def test_wisdom_score_wisdom_level(self):
        ws = WisdomStore()
        ws.register("m1", confidence=0.85, salience=0.75)
        ws.corroborate("m1")
        ws.corroborate("m1")
        assert ws.wisdom_score("m1") == pytest.approx(1.0)

    def test_wisdom_score_unknown_id(self):
        ws = WisdomStore()
        assert ws.wisdom_score("ghost") == 0.0


class TestWisdomStoreManualPromotion:
    def test_manual_promote_to_wisdom(self):
        ws = WisdomStore()
        ws.register("m1", confidence=0.1)
        ws.promote("m1", WisdomLevel.WISDOM, notes="manually curated")
        assert ws.is_wisdom("m1")
        assert ws._entries["m1"].notes == "manually curated"

    def test_manual_promote_unknown_id_no_error(self):
        ws = WisdomStore()
        ws.promote("ghost", WisdomLevel.WISDOM)


class TestWisdomStoreCount:
    def test_count_by_level(self):
        ws = WisdomStore()
        ws.register("m1", confidence=0.1)
        ws.register("m2", confidence=0.45)
        ws.register("m3", confidence=0.85, salience=0.75)
        ws.corroborate("m3")
        ws.corroborate("m3")
        counts = ws.count()
        assert counts["raw"] >= 1
        assert counts["processed"] >= 1
        assert counts["wisdom"] >= 1

    def test_list_wisdom_returns_only_wisdom(self):
        ws = WisdomStore()
        ws.register("m1", confidence=0.1)
        ws.register("m2", confidence=0.85, salience=0.75)
        ws.corroborate("m2")
        ws.corroborate("m2")
        wisdom_entries = ws.list_wisdom()
        assert all(e.level == WisdomLevel.WISDOM for e in wisdom_entries)
