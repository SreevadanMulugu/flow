#!/usr/bin/env python3
"""
Export Parakeet TDT 0.6B v3 from NeMo to ONNX + INT8 quantization.

Run this ONCE on a machine with NeMo installed.
Output goes to ~/.flow/models/parakeet-onnx/

Usage:
    # In the NeMo venv (Python 3.10+):
    python scripts/export_onnx.py

    # With INT8 quantization (smaller, faster on CPU):
    python scripts/export_onnx.py --quantize

    # Custom output dir:
    python scripts/export_onnx.py --output /path/to/models
"""
import os, sys, argparse
import numpy as np

DEFAULT_OUT = os.path.expanduser("~/.flow/models/parakeet-onnx")

def export(output_dir: str, quantize: bool):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading NeMo model (this downloads ~2.4GB on first run)...")
    import nemo.collections.asr as nemo_asr
    import torch

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model.eval()

    # Save tokenizer (SentencePiece model) for ONNX backend decoding
    tok_dst = os.path.join(output_dir, "tokenizer.model")
    if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "tokenizer"):
        sp_model_path = model.tokenizer.tokenizer.vocab_file
        import shutil
        shutil.copy(sp_model_path, tok_dst)
        print(f"Tokenizer saved → {tok_dst}")
    else:
        print("Warning: Could not extract tokenizer — token decoding may be limited")

    encoder_path = os.path.join(output_dir, "encoder.onnx")
    decoder_path = os.path.join(output_dir, "decoder_joint.onnx")

    print("Exporting encoder...")
    _export_encoder(model, encoder_path)

    print("Exporting decoder/joint network...")
    _export_decoder(model, decoder_path)

    if quantize:
        print("Quantizing to INT8...")
        _quantize(encoder_path, decoder_path, output_dir)

    print(f"\nDone. Models saved to: {output_dir}")
    print("Run Flow — it will auto-detect and use ONNX backend.")

def _export_encoder(model, out_path: str):
    import torch
    from torch.onnx import export as onnx_export

    # Dummy input: [batch=1, n_mels=80, T=300]
    dummy_feat    = torch.randn(1, 80, 300)
    dummy_len     = torch.tensor([300], dtype=torch.long)

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, feat, feat_len):
            out, out_len = self.enc(audio_signal=feat, length=feat_len)
            return out, out_len

    wrapper = EncoderWrapper(model.encoder)
    wrapper.eval()

    with torch.no_grad():
        onnx_export(
            wrapper,
            (dummy_feat, dummy_len),
            f=out_path,
            input_names=["features", "feature_lengths"],
            output_names=["encoder_output", "encoder_lengths"],
            dynamic_axes={
                "features":        {2: "time"},
                "encoder_output":  {1: "time"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"  Encoder → {out_path}  ({os.path.getsize(out_path) // 1024 // 1024} MB)")

def _export_decoder(model, out_path: str):
    import torch
    from torch.onnx import export as onnx_export

    # Decoder joint: encoder frame [1,1,D] + prev token [1,1] + hidden → logits + hidden
    D = model.encoder._feat_out if hasattr(model.encoder, "_feat_out") else 512

    dummy_enc   = torch.randn(1, 1, D)
    dummy_token = torch.tensor([[0]], dtype=torch.long)

    # Try to get hidden state dimension from decoder
    try:
        dummy_h = torch.zeros(1, 1, model.decoder.hidden_size)
    except Exception:
        dummy_h = torch.zeros(1, 1, 640)

    class DecoderJointWrapper(torch.nn.Module):
        def __init__(self, dec, joint):
            super().__init__()
            self.dec   = dec
            self.joint = joint
        def forward(self, enc_out, prev_token, h_prev):
            # Embedding
            emb = self.dec.prediction.embed(prev_token)
            # RNN step
            rnn_out, h_new = self.dec.prediction.rnn(emb, h_prev)
            # Joint
            logits = self.joint.joint_net(
                torch.cat([enc_out, rnn_out], dim=-1)
            )
            return logits, h_new

    try:
        wrapper = DecoderJointWrapper(model.decoder, model.joint)
        wrapper.eval()
        with torch.no_grad():
            onnx_export(
                wrapper,
                (dummy_enc, dummy_token, dummy_h),
                f=out_path,
                input_names=["encoder_frame", "prev_token", "hidden_in"],
                output_names=["logits", "hidden_out"],
                dynamic_axes={},
                opset_version=17,
                do_constant_folding=True,
            )
        print(f"  Decoder → {out_path}  ({os.path.getsize(out_path) // 1024 // 1024} MB)")
    except Exception as e:
        print(f"  Decoder export failed: {e}")
        print("  Falling back to full-model batch export (no streaming decoder)...")
        _export_full_model_fallback(model, out_path)

def _export_full_model_fallback(model, out_path: str):
    """Export full model via NeMo's built-in export() — less optimal but reliable."""
    try:
        model.export(out_path)
        print(f"  Full model → {out_path}")
    except Exception as e:
        print(f"  Full model export also failed: {e}")
        print("  Manual ONNX export may be required for this NeMo version.")

def _quantize(encoder_path: str, decoder_path: str, output_dir: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx

    for src_path, name in [(encoder_path, "encoder"), (decoder_path, "decoder_joint")]:
        if not os.path.exists(src_path):
            continue
        q_path = os.path.join(output_dir, f"{name}_int8.onnx")
        quantize_dynamic(
            model_input=src_path,
            model_output=q_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
        )
        orig_mb = os.path.getsize(src_path) // 1024 // 1024
        q_mb    = os.path.getsize(q_path) // 1024 // 1024
        print(f"  {name}: {orig_mb}MB → {q_mb}MB INT8")

        # Replace originals with quantized versions
        os.replace(q_path, src_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--quantize", action="store_true",  help="Apply INT8 quantization")
    args = parser.parse_args()
    export(args.output, args.quantize)
