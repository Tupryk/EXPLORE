from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def warmup_cos_scheduler(optimizer, warmup_epochs: int, total_epochs: int) -> SequentialLR:
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )
