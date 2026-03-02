from examples.synthetic_selective_observation import (
    argmin,
    empirical_objective_for_theta,
    latent_objective_for_theta,
    robust_objective_for_theta,
)


def test_robust_solution_moves_toward_latent_full_data_optimum() -> None:
    erm_theta, _ = argmin(empirical_objective_for_theta)
    robust_theta, _ = argmin(robust_objective_for_theta)
    latent_theta, _ = argmin(latent_objective_for_theta)

    assert erm_theta < robust_theta <= latent_theta
