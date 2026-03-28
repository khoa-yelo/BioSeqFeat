import argparse
import dgeb
from random_baseline import RandomBioSeqTransformer
import os

os.environ["HF_HOME"] = "/scratch/users/khoang99/hf_cache"
import multiprocessing as mp

mp.set_start_method("fork", force=True)
#    "facebook/esm2_t6_8M_UR50D",
#     "Rostlab/prot_t5_xl_uniref50",
#    "facebook/esm2_t30_150M_UR50D",
#     "hugohrban/progen2-medium",

HF_MODELS = [
    "facebook/esm2_t33_650M_UR50D",
]

TASKS = dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN)


def run_hf_model(model_name: str):
    model = dgeb.get_model(model_name)
    evaluation = dgeb.DGEB(tasks=TASKS)
    output_folder = f"results/{model_name.replace('/', '_')}"
    results = evaluation.run(model, output_folder=output_folder)
    return results


def run_random_baseline(embed_dim: int = 128, num_layers: int = 2):
    model = RandomBioSeqTransformer(embed_dim=embed_dim, num_layers=num_layers)
    evaluation = dgeb.DGEB(tasks=TASKS)
    results = evaluation.run(model, output_folder="results/random_baseline")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run DGEB evaluation on one or more models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=HF_MODELS,
        help="HuggingFace model names to evaluate (default: all in HF_MODELS list)",
    )
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Also run the random baseline model",
    )
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dim for random baseline")
    parser.add_argument("--num-layers", type=int, default=2, help="Num layers for random baseline")
    args = parser.parse_args()

    all_results = {}

    for model_name in args.models:
        print(f"\n=== Running {model_name} ===")
        all_results[model_name] = run_hf_model(model_name)

    if args.random_baseline:
        print("\n=== Running random baseline ===")
        all_results["random_baseline"] = run_random_baseline(
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
        )

    return all_results


if __name__ == "__main__":
    main()
