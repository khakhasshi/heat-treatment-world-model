import numpy as np
import torch

from heat_pinn.model import HeatPINN, heat_equation_residual
from heat_pinn.problem import HeatEquation1D


def test_exact_solution_satisfies_initial_and_boundary_conditions() -> None:
    problem = HeatEquation1D()
    x = np.linspace(0.0, 1.0, 51)
    initial = problem.exact_dimensionless_numpy(x, np.zeros_like(x))
    np.testing.assert_allclose(initial, np.sin(np.pi * x), atol=1e-12)

    t = np.linspace(0.0, 1.0, 51)
    np.testing.assert_allclose(
        problem.exact_dimensionless_numpy(np.zeros_like(t), t), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        problem.exact_dimensionless_numpy(np.ones_like(t), t), 0.0, atol=1e-12
    )


def test_exact_solution_has_small_autodiff_pde_residual() -> None:
    problem = HeatEquation1D()

    class ExactModel(torch.nn.Module):
        def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
            return problem.exact_dimensionless_torch(
                coordinates[:, 0:1], coordinates[:, 1:2]
            )

    coordinates = torch.rand(64, 2)
    residual = heat_equation_residual(ExactModel(), coordinates, problem.alpha)
    assert float(torch.max(torch.abs(residual)).detach()) < 1e-5


def test_network_output_shape() -> None:
    model = HeatPINN((8, 8))
    assert model(torch.rand(17, 2)).shape == (17, 1)
