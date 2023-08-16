import typer

from plearning.best_epochs import best_epochs
from plearning.data import create_partitions, create_segments, process_audio, vad, verify
from plearning.evaluate import evaluate_cpc_best_epochs, evaluate_cpc_pre_computed, evaluate_mfcc
from plearning.mfcc import compute_mfcc
from plearning.pairs import abx_pairs
from plearning.scores import recap_scores
from plearning.train import launch_training
from plearning.tsne import build_tsne

data = typer.Typer(help="Data processing utilities", pretty_exceptions_enable=False)
data.command(name="partition")(create_partitions)
data.command(name="segment")(create_segments)
data.command(name="remix")(process_audio)
data.command(name="vad")(vad)
data.command(name="verify")(verify)

evaluate = typer.Typer(help="Create jobs to evaluate models on ABX tasks", pretty_exceptions_enable=False)
evaluate.command(name="best")(evaluate_cpc_best_epochs)
evaluate.command(name="features")(evaluate_cpc_pre_computed)
evaluate.command(name="mfcc")(evaluate_mfcc)

main = typer.Typer(help="Perceptual narrowing with CPC", pretty_exceptions_enable=False)
main.add_typer(data, name="data")
main.add_typer(evaluate, name="evaluate")
main.command(name="train")(launch_training)
main.command(name="pairs")(abx_pairs)
main.command(name="scores")(recap_scores)
main.command(name="mfcc")(compute_mfcc)
main.command(name="best_epochs")(best_epochs)
main.command(name="tsne")(build_tsne)

if __name__ == "__main__":
    main()
