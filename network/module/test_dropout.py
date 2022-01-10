import unittest
import torch
from dropout import *

class TestDropout(unittest.TestCase):

    def setUp(self) -> None:
        # All of these tests are slightly randomized
        torch.manual_seed(0)
        return super().setUp()

    def test_dropout_scaling(self):
        """
        Dropout must be scaled such that the testing network has similar weights
        to the thinned networks.
        """
        n = 100_000
        input = torch.ones(n)
        probability = torch.rand(n)
        self.assertAlmostEqual(
            dropout(input, probability, True).sum()/n, 1.0, delta=0.05)

    def test_dropout(self):
        """
        Test dropout will drop the correct number of elements
        """
        n = 100_000
        input = torch.ones(n)
        probability = torch.rand(1)
        self.assertAlmostEqual(
            dropout(input, torch.ones(n)*probability, True).count_nonzero()/n, 
            (1-probability), delta=0.05)

    def test_test_mode(self):
        """Test mode should leave the output unchanged"""
        input = torch.rand(10)
        self.assertTrue(torch.equal(dropout(input, torch.ones(10), False), input))

    def test_conditioned(self):
        n = 100_000
        groups = 5
        p_active = 0.1
        p_inactive = 0.8
        cond_dropout = ConditionedDropout(n, groups, p_active, p_inactive)

        # Approximate Group Size
        group_size = n/groups

        # Verify that they are assigned to roughly equal groups
        for i in range(groups):
            self.assertAlmostEqual(cond_dropout.group_ids.eq(i).sum()/group_size, 1.0, delta=0.02)

        output = cond_dropout.forward(torch.ones(n))

        # Frequency of active units in the active group
        mask = cond_dropout.group_ids.eq(cond_dropout.active_group)
        freq_active = (output * mask).count_nonzero()
        freq_inactive = (output * ~mask).count_nonzero()

        # Assert that the expected number of active units matches the actual when n is large
        self.assertAlmostEqual(float(freq_active/group_size), 1-p_active, delta=0.01)
        self.assertAlmostEqual(float(freq_inactive)/(group_size*(groups-1)), 1-p_inactive, delta=0.01)



if __name__ == '__main__':
    unittest.main()