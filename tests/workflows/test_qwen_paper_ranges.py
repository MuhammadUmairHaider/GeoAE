"""The Qwen appendix preserves held-out discovery and exact nonlinear edit semantics."""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from geoae.model import GeoAE


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / "experiments/nonlinear_decoders/deltaai/qwen_paper_ranges.py"
    spec = importlib.util.spec_from_file_location("qwen_paper_ranges", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_streaming_range_discovery_finds_parallel_separated_clusters(runner):
    jitter = torch.linspace(-0.1, 0.1, 12)
    values = torch.cat([torch.stack([jitter + center, torch.zeros_like(jitter)], 1)
                        for center in (1.0, 3.0, -4.0)])
    labels = torch.arange(3).repeat_interleave(12)

    def batches():
        for start in range(0, len(values), 7):
            yield {"geoae": (values[start:start + 7], labels[start:start + 7])}

    result = runner.count_range_pairs(batches, n_clusters=4, min_n=10)
    assert result["meta"]["rows"] == 36
    assert result["summary"]["geoae"]["range_pairs"] == 1
    assert result["summary"]["geoae"]["clusters_with_partner"] == 3
    pair = result["range_pairs"]["geoae"][0]
    assert (pair["a"], pair["b"]) == (0, 1)
    assert pair["cos"] == pytest.approx(1.0)
    assert pair["dprime"] == pytest.approx(2 / jitter.std(correction=0).item(), rel=1e-5)
    assert pair["n_a"] == pair["n_b"] == 12


def test_discovery_reports_zero_when_minimum_cluster_support_is_not_met(runner):
    def batches():
        yield {"geoae": (torch.tensor([[1.0, 0.0], [2.0, 0.0]]), torch.tensor([0, 1]))}

    result = runner.count_range_pairs(batches, 2, min_n=50)
    assert result["range_pairs"]["geoae"] == []
    assert result["summary"]["geoae"]["range_pairs"] == 0
    assert result["summary"]["geoae"]["range_pairs_share_of_pairs"] == 0


@pytest.mark.parametrize("kind", ["mlp", "residual_mlp"])
@torch.no_grad()
def test_nonlinear_delta_preserves_original_residual_and_cancels_biases(runner, kind):
    ae = GeoAE(hidden_size=2, latent_dim=2, n_clusters=2, nonlinearity="linear",
               decoder_type=kind, decoder_hidden_dim=2, decoder_output_bias=True,
               token_bias_rows=2, vocab_size=2).eval()
    hidden = ae.decoder[0] if kind == "mlp" else ae.decoder.hidden
    output = ae.decoder[2] if kind == "mlp" else ae.decoder.output
    hidden.weight.copy_(torch.eye(2)); hidden.bias.zero_()
    output.weight.copy_(torch.eye(2))
    if kind == "mlp":
        output.bias.fill_(50.0)
    else:
        ae.decoder.skip.weight.copy_(torch.eye(2)); ae.decoder.skip.bias.fill_(50.0)
    ae.load_token_bias(torch.tensor([0, 1]), torch.tensor([[10.0, -20.0], [-8.0, 9.0]]))
    z = torch.tensor([[-2.0, 1.0], [2.0, -1.0], [-0.5, 0.3]])
    edited = z + torch.tensor([[0.5, 0.25], [0.5, 0.25], [0.0, 0.0]])
    h = torch.tensor([[11.0, 20.0], [30.0, 7.0], [4.0, 2.0]])
    std = torch.tensor([0.5, 3.0])
    result, changed = runner.apply_latent_delta(h, z, edited, torch.tensor([0, 1, 0]), ae, std, chunk=1)
    change = F.gelu(edited) - F.gelu(z)
    if kind == "residual_mlp":
        change += edited - z
    torch.testing.assert_close(result, h + change * std, rtol=2e-5, atol=2e-5)
    assert changed.tolist() == [True, True, False]
    assert torch.equal(result[2], h[2])
    assert not torch.allclose(change[0], change[1])


@torch.no_grad()
def test_loss_scorer_matches_clean_to_edited_kl_and_leaves_untouched_rows_exact_zero(runner):
    head = torch.nn.Linear(2, 3, bias=False)
    head.weight.copy_(torch.tensor([[1.0, 0.2], [-0.3, 1.0], [-1.0, -1.0]]))
    scorer = runner.Scorer(torch.nn.Identity(), head)
    h = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edited = h + 0.2
    nxt = torch.tensor([0, 1])
    nll, kl = scorer.row_metrics(h, edited, torch.tensor([True, False]), nxt)
    clean_lp = head(h).log_softmax(-1)
    edited_lp = head(edited).log_softmax(-1)
    assert nll[0].item() == pytest.approx((clean_lp[0, 0] - edited_lp[0, 0]).item())
    assert kl[0].item() == pytest.approx((clean_lp[0].exp() * (clean_lp[0] - edited_lp[0])).sum().item())
    assert nll[1].item() == kl[1].item() == 0.0


def test_matched_damage_uses_common_reachable_level_and_bootstraps_case_means(runner):
    cases = []
    for multiplier in (1.0, 2.0, 3.0):
        results = {}
        for arm in runner.ARMS:
            slope = 0.1 if arm == "dir_range_z" else 0.8
            results[arm] = {runner.skey(s): {
                "target": {"dnll": s * multiplier, "kl": s, "frac_edited": 1.0},
                "sibling": {"dnll": slope * s * multiplier, "kl": s, "frac_edited": 1.0},
                "random": {"dnll": 0.05 * s * multiplier, "kl": s, "frac_edited": 1.0}}
                for s in runner.GRID[arm]}
        cases.append({"res": results, "T_position": "inner"})
    summary = runner.summarise(cases)
    matched = summary["matched"]
    assert matched["n_cases"] == 3
    assert matched["L_median"] == pytest.approx(0.3)
    assert matched["arms"]["dir_range_z"]["sibling"]["mean"] == pytest.approx(0.03)
    whole = matched["arms"]["dir_all_z"]["sibling"]
    assert whole["mean"] == pytest.approx(0.24)
    assert whole["vs_dir_range_z"]["mean_diff"] == pytest.approx(0.21)
    assert len(whole["mean_ci"]) == 2
    assert whole["mean_ci"][0] <= whole["mean"] <= whole["mean_ci"][1]


def test_cli_has_qwen_protocol_defaults_and_explicit_output_paths(runner, tmp_path):
    args = runner.parse_args(["--checkpoint", "ae.pt", "--activations_dir", "acts", "--km", "raw.npz",
                              "--kmt", "tok.npz", "--token_bias_table", "table.npz", "--out_dir", str(tmp_path)])
    assert args.layer == 35 and args.model_name == "Qwen/Qwen2.5-3B"
    assert args.count_min_n == 50 and args.count_dprime == 2.0
    assert args.min_n == 100 and args.min_dprime == 3.0
    assert args.n_pairs == 30 and args.min_rows == 30
    assert Path(args.ranges).name == "ranges_along_directions.json"
    assert Path(args.out).name == "same_direction_removal.json"
    assert args.no_showcase
