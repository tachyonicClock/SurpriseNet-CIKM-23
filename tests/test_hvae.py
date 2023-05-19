from network.deep_vae import CyclicBeta
import pytest


def test_cyclic_beta():
    cycles = 4
    schedule = iter(CyclicBeta(100, 0.5, cycles))
    cycle_length = 25
    growth_length = cycle_length / 2

    for c in range(4):
        for i in range(cycle_length):
            beta = next(schedule)
            print(f"cycle {c}, step {i}, beta {beta}")
            if i < growth_length:
                assert beta == pytest.approx(i * 1 / growth_length)
            else:
                assert beta == 1.0
