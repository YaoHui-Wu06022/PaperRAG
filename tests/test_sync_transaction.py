from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from services.sync_transaction import recover_pending_sync_operation


class SyncTransactionRecoveryTest(unittest.TestCase):
    @patch("services.sync_transaction.local_cache_store.clear_sync_journal")
    @patch("services.sync_transaction._apply_local_state")
    @patch("services.sync_transaction._apply_remote_operation")
    @patch("services.sync_transaction._sync_plan_from_journal")
    @patch("services.sync_transaction.local_cache_store.load_sync_journal")
    def test_local_pending_only_applies_local(
        self,
        load_sync_journal: Mock,
        sync_plan_from_journal: Mock,
        apply_remote_operation: Mock,
        apply_local_state: Mock,
        clear_sync_journal: Mock,
    ) -> None:
        payload = {"status": "local_pending"}
        plan = Mock()
        config = Mock()
        embeddings = Mock()

        load_sync_journal.return_value = payload
        sync_plan_from_journal.return_value = plan

        status = recover_pending_sync_operation(config, embeddings)

        self.assertEqual(status, "local_pending")
        apply_remote_operation.assert_not_called()
        apply_local_state.assert_called_once_with(config, plan)
        clear_sync_journal.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()
