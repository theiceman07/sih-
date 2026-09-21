"""
SRM (land-cover fraction) loss: per-pixel soft cross-entropy against
WorldCover-derived fractions, plus the classical sub-pixel mapping
abundance-sum constraint. See FOUR_DAY_PLAN.md section 2B.

The abundance-sum constraint is redundant by construction here (the model's
softmax head in models/srm_head.py already guarantees fractions sum to 1 at
every fine pixel), but it is kept as an explicit, checkable loss term rather
than relied upon implicitly: it is the term that would catch a future head
that outputs unnormalized fractions (e.g. a sigmoid-per-class variant), and
its value or otherwise being ~0 is a printable proof rather than an
assumption -- cheap insurance against a silent architecture change breaking
this specific correctness property.
"""
import torch
import torch.nn as nn


class SRMLoss(nn.Module):
    """
    pred_fractions: (B, n_classes, H, W), softmax output, sums to 1 per pixel.
    target_fractions: (B, n_classes, H, W), same shape, from
        data.worldcover_fetch.labels_to_fractions (also sums to 1 per pixel;
        equals a one-hot vector at hard-labeled coarse pixels, otherwise the
        real sub-pixel class mixture within that coarse cell).

    Loss = soft cross-entropy (-sum target*log(pred)) + lambda_sum * abundance
    -sum-constraint penalty (mean squared deviation of pred's per-pixel sum
    from 1).
    """

    def __init__(self, lambda_sum=0.1, eps=1e-8):
        super().__init__()
        self.lambda_sum = lambda_sum
        self.eps = eps

    def forward(self, pred_fractions, target_fractions):
        ce = -(target_fractions * torch.log(pred_fractions + self.eps)).sum(dim=1).mean()

        pred_sum = pred_fractions.sum(dim=1)
        sum_constraint = ((pred_sum - 1.0) ** 2).mean()

        return ce + self.lambda_sum * sum_constraint


if __name__ == "__main__":
    loss_fn = SRMLoss()

    # Perfect match -> loss close to 0 (soft CE floor is the target's own
    # entropy, not exactly 0, unless target is one-hot).
    target = torch.zeros(2, 4, 8, 8)
    target[:, 0] = 1.0  # one-hot everywhere
    pred = target.clone()
    loss_perfect = loss_fn(pred, target)
    assert loss_perfect.item() < 1e-4, f"expected ~0 for perfect one-hot match, got {loss_perfect.item()}"

    # Uniform random softmax prediction against one-hot target -> higher loss.
    logits = torch.rand(2, 4, 8, 8)
    pred_random = torch.softmax(logits, dim=1)
    loss_random = loss_fn(pred_random, target)
    assert loss_random.item() > loss_perfect.item()

    # Sum-constraint term explicitly: an already-normalized softmax should
    # contribute ~0 sum-constraint penalty on its own.
    pred_sum = pred_random.sum(dim=1)
    assert torch.allclose(pred_sum, torch.ones_like(pred_sum), atol=1e-5)

    print(f"SRMLoss(perfect) = {loss_perfect.item():.6f}")
    print(f"SRMLoss(random)  = {loss_random.item():.4f}")
    print("Sanity checks passed.")
