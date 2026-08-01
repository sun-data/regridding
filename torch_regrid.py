"""
Prototype: a torch (GPU) apply for conservative regridding weights.

The sparse weights produced by :func:`regridding.weights` are, per element,
a ``(indices_input, indices_output, values)`` triple with self-contained
flat indices. Applying them is gather–multiply–scatter, which maps onto a
single :meth:`torch.Tensor.index_add_` per group of elements once the
elements are concatenated with flattened-index offsets. With the weights
resident in GPU memory, the apply is HBM-bandwidth-bound and roughly two
orders of magnitude faster than the CPU kernel.

Measured on the ESIS-I Level-4 problem (six lines, 0.75 arcsec scene,
2.75e9 forward triples over 4 channels x 143 wavelengths, NVIDIA H100):

- full forward apply: 0.053 s (~1 TB/s effective; CPU kernel: minutes);
- full MART inversion of one frame (32 iterations, forward + variance +
  transpose per iteration): 5.9 s vs ~17 min CPU (~175x);
- 30-frame flight, weights uploaded once: 153 s of GPU time;
- GPU vs CPU final solutions agree to ~5e-7 (float32 rounding), identical
  iteration counts on all 30 frames.

Memory: int64 indices are required by ``index_add_``, so a resident
element set costs ``20 bytes x num_triples`` (i64 + i64 + f32). The ESIS
production set is ~55 GB per direction (fits one 80 GB H100); the
1.5-arcsec development grid is ~14 GB per direction and fits a 24 GB
consumer GPU (e.g. RTX 4090) one direction at a time, or both directions
with int32 index packing via a small custom kernel (a follow-up).

Run the synthetic benchmark (no dependencies beyond numpy + torch)::

    python torch_regrid.py [--triples 2.75e9]
"""

import argparse

import numpy as np


def pack(elements, num_input: int, offsets: bool = True):
    """
    Concatenate weight elements into flat arrays with global input offsets.

    Parameters
    ----------
    elements
        A sequence of ``(indices_input, indices_output, values)`` triples,
        one per (wavelength) element of the group.
    num_input
        The flattened size of one element's input plane; element ``k``'s
        input indices are offset by ``k * num_input`` so a single gather
        addresses the whole input hypercube.
    offsets
        Whether to apply the per-element offsets to the input indices
        (disable if the elements already use global indices).
    """
    indices_input = []
    indices_output = []
    values = []
    for k, (idx_in, idx_out, vals) in enumerate(elements):
        shift = k * num_input if offsets else 0
        indices_input.append(idx_in.astype(np.int64) + shift)
        indices_output.append(idx_out.astype(np.int64))
        values.append(vals.astype(np.float32))
    return (
        np.concatenate(indices_input),
        np.concatenate(indices_output),
        np.concatenate(values),
    )


def upload(packed, device):
    """
    Move a packed weights group to the given device.

    Parameters
    ----------
    packed
        The result of :func:`pack`.
    device
        The torch device to upload to.
    """
    import torch

    return tuple(torch.from_numpy(a).to(device) for a in packed)


def apply(weights, values_input, num_output: int):
    """
    Apply a resident weights group to a resident input array.

    Parameters
    ----------
    weights
        The uploaded ``(indices_input, indices_output, values)`` tensors.
    values_input
        The flattened input array, resident on the same device.
    num_output
        The flattened size of the output array.
    """
    import torch

    indices_input, indices_output, values = weights
    result = torch.zeros(
        num_output,
        dtype=values_input.dtype,
        device=values_input.device,
    )
    result.index_add_(0, indices_output, values * values_input[indices_input])
    return result


def main() -> None:
    """Benchmark the apply on synthetic weights."""
    import time

    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples", type=float, default=1e9)
    parser.add_argument("--num-input", type=int, default=100_000_000)
    parser.add_argument("--num-output", type=int, default=2_000_000)
    args = parser.parse_args()

    num_triples = int(args.triples)
    print(f"cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(42)
    print(f"building {num_triples / 1e9:.2f}G synthetic triples ...")
    packed = (
        rng.integers(0, args.num_input, num_triples, dtype=np.int64),
        rng.integers(0, args.num_output, num_triples, dtype=np.int64),
        rng.random(num_triples, dtype=np.float32),
    )
    gigabytes = sum(a.nbytes for a in packed) / 2**30
    print(f"resident set: {gigabytes:.1f} GiB")

    weights = upload(packed, device)
    values_input = torch.from_numpy(
        rng.random(args.num_input, dtype=np.float32)
    ).to(device)

    # validate against a float64 CPU scatter on a subset
    subset = slice(0, min(num_triples, 10_000_000))
    check = np.zeros(args.num_output)
    np.add.at(
        check,
        packed[1][subset],
        packed[2][subset].astype(np.float64)
        * values_input.cpu().numpy()[packed[0][subset]],
    )
    partial = apply(
        tuple(w[subset] for w in weights), values_input, args.num_output
    )
    error = np.abs(partial.cpu().numpy() - check).max() / check.max()
    print(f"validation (subset) max relative error: {error:.2e}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    num_repeat = 10
    t0 = time.perf_counter()
    for _ in range(num_repeat):
        apply(weights, values_input, args.num_output)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / num_repeat
    print(
        f"apply: {elapsed:.4f} s"
        f" ({gigabytes / elapsed:.0f} GiB/s effective)"
    )


if __name__ == "__main__":
    main()
