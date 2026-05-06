"""
Standalone MossFormer2 separation runner — called as a subprocess.
Usage: python mossformer2_run.py <input_wav> <out_dir> <num_speakers>
Writes: <out_dir>/speaker_1.wav, speaker_2.wav, ...
Exit 0 = success, non-zero = failure.
"""
import sys, os

def main():
    if len(sys.argv) < 4:
        print("Usage: mossformer2_run.py <input> <out_dir> <num_spks>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    out_dir    = sys.argv[2]
    num_spks   = int(sys.argv[3])

    os.makedirs(out_dir, exist_ok=True)

    # Change to project root so clearvoice downloads checkpoints to
    # <project>/checkpoints/MossFormer2_SS_16K
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    import warnings
    warnings.filterwarnings("ignore")

    from clearvoice import ClearVoice
    import numpy as np
    import soundfile as sf

    print("[MossFormer2] Loading MossFormer2_SS_16K…", file=sys.stderr)
    cv = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])
    print("[MossFormer2] Model loaded. Separating…", file=sys.stderr)

    result = cv(input_path=input_path, online_write=False)

    # result is dict keyed by model name, value is list of numpy arrays (one per speaker)
    if isinstance(result, dict):
        streams = list(result.values())[0]
    elif isinstance(result, (list, np.ndarray)):
        streams = result if isinstance(result, list) else [result]
    else:
        print(f"[MossFormer2] Unexpected result type: {type(result)}", file=sys.stderr)
        sys.exit(2)

    n_out = min(num_spks, len(streams))
    sr = 16000
    for i in range(n_out):
        src = np.array(streams[i], dtype=np.float32)
        if src.ndim > 1:
            src = src.mean(axis=0)
        peak = np.abs(src).max()
        if peak > 0:
            src = src / peak * 0.9
        out_path = os.path.join(out_dir, f"speaker_{i+1}.wav")
        sf.write(out_path, src, sr)
        print(f"[MossFormer2] Wrote {out_path}", file=sys.stderr)

    print(f"[MossFormer2] Done — {n_out} speakers separated.", file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    main()
