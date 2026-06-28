"""Underfit gradio launcher.

Backend-agnostic wrapper. Selects between the sat and sa3 backends and
delegates UI construction to the backend's create_gradio_ui().

CLI is the same as SAT-dev's run_gradio.py so dashboard launches don't need
to change beyond pointing at this script.
"""
import sys
import warnings

# Always-on suppression of two torchaudio UserWarnings that fire on every
# inference call (the SA3 pretransform's mel-spec is reconstructed per call,
# so the warnings re-emit each generation). These two are specifically known
# noise; the broader filterwarnings("ignore") below covers the rest unless
# --verbose was passed. Done as targeted filters so they survive any library
# that calls warnings.resetwarnings() during init.
warnings.filterwarnings(
    "ignore",
    message=r".*'onesided' has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*At least one mel filterbank has all zero values.*",
    category=UserWarning,
)

if "--verbose" not in sys.argv:
    import os as _os
    _os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore")

import argparse
import os
import struct
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

import torch

from underfit.backends import get_backend


# ── stable_audio_demos output helpers ────────────────────────────────────────

def _get_demos_dir() -> Path:
    """Return (and create) stable_audio_demos/ next to the underfit package root."""
    d = Path(__file__).parent.parent / "stable_audio_demos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _uk_timestamp() -> str:
    """Current UK local time as a filename-safe string."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/London")
    except Exception:
        tz = timezone.utc
    return datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")


def _unique_path(base: Path) -> Path:
    """Return base if free, else base_1, base_2 … until one is free."""
    if not base.exists():
        return base
    n = 1
    while True:
        candidate = base.parent / f"{base.stem}_{n}{base.suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _write_wav_comment(wav_path: Path, comment: str):
    """Embed text into a WAV's RIFF LIST INFO ICMT (Comment) chunk, in-place."""
    try:
        data = wav_path.read_bytes()
        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return
        enc = comment[:500].encode("latin-1", errors="replace") + b"\x00"
        if len(enc) % 2:
            enc += b"\x00"
        icmt = b"ICMT" + struct.pack("<I", len(enc)) + enc
        list_block = b"LIST" + struct.pack("<I", 4 + len(icmt)) + b"INFO" + icmt
        out = bytearray(data[:12])
        i = 12
        while i + 8 <= len(data):
            cid = data[i:i + 4]
            csz = struct.unpack_from("<I", data, i + 4)[0]
            chunk_end = i + 8 + csz + (csz % 2)
            if cid == b"LIST" and i + 12 <= len(data) and data[i + 8:i + 12] == b"INFO":
                i = chunk_end
                continue
            out += data[i:min(chunk_end, len(data))]
            i = chunk_end
        out += list_block
        struct.pack_into("<I", out, 4, len(out) - 8)
        wav_path.write_bytes(bytes(out))
    except Exception:
        pass


def _extract_path(result) -> Path | None:
    """Pull a filesystem path out of whatever Gradio's postprocess returns."""
    if result is None:
        return None
    if isinstance(result, dict):
        p = result.get("path") or result.get("name") or result.get("value")
    elif hasattr(result, "path"):
        p = result.path
    elif hasattr(result, "name"):
        p = result.name
    elif isinstance(result, str):
        p = result
    else:
        p = None
    return Path(p) if p else None


def _install_output_hook(interface):
    """Patch gr.Audio and gr.Image postprocess methods so every generated WAV
    (and its waveform image) is copied into stable_audio_demos/ with a UK
    timestamp filename and the prompt written into the WAV Comment tag.

    We COPY rather than move so Gradio's internal file-server keeps serving
    the original path to the UI player — no 404s, no Windows file-lock errors.

    Audio and Image outputs are paired by a shared timestamp stored in a
    thread-local, so each waveform .webp lands alongside its .wav.
    """
    try:
        import gradio as gr
    except ImportError:
        print("[demos] gradio not importable — output hook skipped", flush=True)
        return

    demos_dir = _get_demos_dir()

    def _make_audio_hook(original_pp):
        def _hooked(y, *a, **kw):
            result = original_pp(y, *a, **kw)
            try:
                src = _extract_path(result)
                if src and src.exists() and src.suffix.lower() == ".wav":
                    stamp = _uk_timestamp()
                    dst = _unique_path(demos_dir / f"{stamp}.wav")
                    shutil.copy2(str(src), str(dst))
                    prompt = src.stem.replace("_", " ").strip()
                    _write_wav_comment(dst, prompt)
                    print(f"[demos] WAV:   {dst.name}  prompt: {prompt[:80]}", flush=True)

                    # SA3 writes the waveform webp as a side-effect alongside
                    # the wav, not through a Gradio component — so we look for
                    # any image file with the same stem. Wait briefly in case
                    # SA3 finishes writing it slightly after the wav.
                    def _generate_spectrogram(src_wav, dst_wav):
                        """Generate a labelled mel spectrogram (title, axis labels,
                        dB colourbar) matching SA3's MelSpectrogram output style,
                        saved as .webp alongside dst_wav."""
                        try:
                            import numpy as np
                            import wave
                            import matplotlib
                            matplotlib.use("Agg")   # headless — must be set before pyplot import
                            import matplotlib.pyplot as plt

                            # ── 1. Read WAV ───────────────────────────────────
                            with wave.open(str(src_wav), "rb") as wf:
                                n_channels  = wf.getnchannels()
                                sampwidth   = wf.getsampwidth()
                                sample_rate = wf.getframerate()
                                raw         = wf.readframes(wf.getnframes())

                            if sampwidth == 2:
                                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                            elif sampwidth == 4:
                                samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                            elif sampwidth == 3:
                                b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                                s = (b[:, 0].astype(np.int32)
                                     | (b[:, 1].astype(np.int32) << 8)
                                     | (b[:, 2].astype(np.int32) << 16))
                                s = np.where(s >= 0x800000, s - 0x1000000, s)
                                samples = s.astype(np.float32) / 8388608.0
                            else:
                                samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

                            if n_channels > 1:
                                samples = samples.reshape(-1, n_channels).mean(axis=1)

                            # ── 2. STFT ───────────────────────────────────────
                            n_fft, hop, n_mels = 2048, 512, 128
                            window  = np.hanning(n_fft)
                            samples = np.pad(samples, (n_fft // 2, n_fft // 2))
                            frames  = 1 + (len(samples) - n_fft) // hop
                            stft    = np.zeros((n_fft // 2 + 1, frames), dtype=np.float32)
                            for i in range(frames):
                                seg = samples[i * hop: i * hop + n_fft] * window
                                stft[:, i] = np.abs(np.fft.rfft(seg, n=n_fft)).astype(np.float32)

                            # ── 3. Mel filterbank ─────────────────────────────
                            def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
                            def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
                            f_max   = float(sample_rate) / 2.0
                            mel_pts = np.linspace(hz_to_mel(0.0), hz_to_mel(f_max), n_mels + 2)
                            bin_pts = np.floor((n_fft + 1) * mel_to_hz(mel_pts) / sample_rate).astype(int)
                            fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
                            for m in range(1, n_mels + 1):
                                lo, cen, hi = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
                                if cen > lo:
                                    for k in range(lo, cen):
                                        fb[m-1, k] = (k - lo) / (cen - lo)
                                if hi > cen:
                                    for k in range(cen, hi):
                                        fb[m-1, k] = (hi - k) / (hi - cen)

                            mel_spec = fb @ stft
                            mel_db   = 10.0 * np.log10(mel_spec ** 2 + 1e-9)
                            mel_db   = np.clip(mel_db, mel_db.max() - 80.0, mel_db.max())

                            # ── 4. Plot matching SA3 style ────────────────────
                            fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=100)
                            im   = ax.imshow(mel_db, aspect="auto", origin="lower",
                                             cmap="viridis", interpolation="nearest")
                            cbar = fig.colorbar(im, ax=ax)
                            cbar.ax.tick_params(labelsize=8)
                            ax.set_title("MelSpectrogram", fontsize=11)
                            ax.set_xlabel("frame", fontsize=9)
                            ax.set_ylabel("mel bins (log freq)", fontsize=9)
                            ax.tick_params(labelsize=8)
                            fig.tight_layout()

                            # ── 5. Save as webp ───────────────────────────────
                            webp_dst = dst_wav.with_suffix(".webp")
                            fig.savefig(str(webp_dst), format="webp", dpi=100,
                                        bbox_inches="tight")
                            plt.close(fig)
                            print(f"[demos] Spectrogram: {webp_dst.name}", flush=True)

                        except Exception as e:
                            print(f"[demos] Spectrogram generation failed: {e}", flush=True)

                    threading.Thread(
                        target=_generate_spectrogram,
                        args=(src, dst),
                        daemon=True,
                    ).start()

            except Exception as e:
                print(f"[demos] Audio hook error: {e}", flush=True)
            return result
        return _hooked

    # Walk all components on the interface
    try:
        blocks = getattr(interface, "blocks", {})
        components = list(blocks.values()) if isinstance(blocks, dict) else []
        if not components:
            components = getattr(interface, "components", [])
    except Exception:
        components = []

    audio_hooked = 0
    for comp in components:
        if isinstance(comp, gr.Audio) and hasattr(comp, "postprocess"):
            comp.postprocess = _make_audio_hook(comp.postprocess)
            audio_hooked += 1

    print(f"[demos] Hooks installed — {audio_hooked} Audio component(s) → {demos_dir}", flush=True)
    if not audio_hooked:
        print("[demos] Warning: no gr.Audio components found — check SA3 backend version", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    backend = get_backend(args.backend)
    print(f"Using backend: {backend.NAME}", flush=True)

    try:
        from stable_audio_tools.verbose import set_verbose
        set_verbose(args.verbose)
    except ImportError:
        pass

    torch.manual_seed(42)

    interface = backend.create_gradio_ui(
        model_config_path=args.model_config,
        ckpt_path=args.ckpt_path,
        pretrained_name=args.pretrained_name,
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        model_half=args.model_half,
        gradio_title=args.title,
        lora_ckpt_paths=args.lora_ckpt_path,
        default_prompt=args.default_prompt,
    )

    # Install copy-on-generate hooks before launch
    _install_output_hook(interface)

    interface.queue()
    interface.launch(
        share=False,
        auth=(args.username, args.password) if args.username is not None else None,
        js=getattr(interface, "_sao_js", None),
        theme=getattr(interface, "_sao_theme", None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gradio interface (Underfit)")
    parser.add_argument("--backend", default=None, help="sat | sa3 (default: env UNDERFIT_BACKEND or auto)")
    parser.add_argument("--pretrained-name", type=str, required=False)
    parser.add_argument("--model-config", type=str, required=False)
    parser.add_argument("--ckpt-path", type=str, required=False)
    parser.add_argument("--pretransform-ckpt-path", type=str, required=False)
    parser.add_argument("--username", type=str, required=False)
    parser.add_argument("--password", type=str, required=False)
    parser.add_argument("--model-half", action="store_true", default=True)
    parser.add_argument("--title", type=str, required=False)
    parser.add_argument("--lora-ckpt-path", type=str, nargs="*", required=False)
    parser.add_argument("--default-prompt", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    main(args)