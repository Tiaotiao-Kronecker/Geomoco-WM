from __future__ import annotations

import unittest

import torch

from geomoco_wm.data.predicted_event_mixture import select_event_candidates


class PredictedEventMixtureTests(unittest.TestCase):
    def test_select_event_candidates_topk_matches_probability_order(self) -> None:
        event_probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
        classes = (
            "sustain_open::none",
            "transition_close::early",
            "transition_open::late",
            "sustain_closed::none",
        )

        top_probs, top_indices = select_event_candidates(
            event_probs,
            classes,
            top_m=2,
            policy="topk",
        )

        self.assertTrue(torch.equal(top_indices, torch.tensor([[0, 1]])))
        self.assertTrue(torch.allclose(top_probs, torch.tensor([[4.0 / 7.0, 3.0 / 7.0]])))

    def test_transition_reserve_replaces_lowest_nontransition_slot(self) -> None:
        event_probs = torch.tensor([[0.40, 0.30, 0.18, 0.12]])
        classes = (
            "sustain_open::none",
            "sustain_closed::none",
            "transition_close::early",
            "transition_open::late",
        )

        top_probs, top_indices = select_event_candidates(
            event_probs,
            classes,
            top_m=2,
            policy="transition_reserve",
            transition_reserve_threshold=0.15,
        )

        self.assertTrue(torch.equal(top_indices, torch.tensor([[0, 2]])))
        self.assertTrue(torch.allclose(top_probs, torch.tensor([[0.40 / 0.58, 0.18 / 0.58]])))

    def test_transition_reserve_keeps_topk_when_transition_already_selected(self) -> None:
        event_probs = torch.tensor([[0.40, 0.35, 0.20, 0.05]])
        classes = (
            "sustain_open::none",
            "transition_close::early",
            "sustain_closed::none",
            "transition_open::late",
        )

        _, top_indices = select_event_candidates(
            event_probs,
            classes,
            top_m=2,
            policy="transition_reserve",
            transition_reserve_threshold=0.15,
        )

        self.assertTrue(torch.equal(top_indices, torch.tensor([[0, 1]])))

    def test_transition_reserve_respects_threshold(self) -> None:
        event_probs = torch.tensor([[0.45, 0.30, 0.14, 0.11]])
        classes = (
            "sustain_open::none",
            "sustain_closed::none",
            "transition_close::early",
            "transition_open::late",
        )

        _, top_indices = select_event_candidates(
            event_probs,
            classes,
            top_m=2,
            policy="transition_reserve",
            transition_reserve_threshold=0.15,
        )

        self.assertTrue(torch.equal(top_indices, torch.tensor([[0, 1]])))


if __name__ == "__main__":
    unittest.main()
