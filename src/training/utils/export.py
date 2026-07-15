"""Export helpers shared by the CNN and ViT training entry points."""

import shutil
from pathlib import Path

import torch


def export_model(
    model_cls: type,
    ckpt_path: str,
    model_path: str | Path,
    sample_batch: tuple[torch.Tensor, ...],
) -> None:
    """Export ``ckpt_path`` to ONNX at ``model_path`` and copy the ckpt next to it.

    Args:
        model_cls: LightningModule class used to reload the checkpoint.
        ckpt_path (str): Path of the checkpoint to export.
        model_path (str | Path): Destination ``.onnx`` path (the model the app
            serves by default). The checkpoint is copied alongside it with the
            same stem and a ``.ckpt`` suffix. Either suffix is accepted.
        sample_batch (tuple[torch.Tensor, ...]): One dataloader batch; its first
            two tensors are used as the (x_units, x_traits) example inputs to
            trace the ONNX graph.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_dest = model_path.with_suffix(".ckpt")
    if Path(ckpt_path).resolve() != ckpt_dest.resolve():
        shutil.copy2(ckpt_path, ckpt_dest)
        print(f"Saved checkpoint -> {ckpt_dest}")

    model = model_cls.load_from_checkpoint(ckpt_path, map_location="cpu").eval()
    x_units, x_traits = (t[:1].cpu() for t in sample_batch[:2])

    onnx_path = model_path.with_suffix(".onnx")
    torch.onnx.export(
        model,
        (x_units, x_traits),
        str(onnx_path),
        input_names=["x_units", "x_traits"],
        output_names=["output"],
        dynamic_axes={
            "x_units": {0: "batch"},
            "x_traits": {0: "batch"},
            "output": {0: "batch"},
        },
        dynamo=False,
    )
    print(f"Saved ONNX model -> {onnx_path}")
