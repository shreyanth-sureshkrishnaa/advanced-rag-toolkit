import json
import datetime
import pathlib
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from pipeline import AdvancedRAG
from scorers import (
    score_faithfulness,
    score_answer_relevance,
    score_context_precision,
    score_ground_truth_similarity,
)

BENCHMARK_PATH = pathlib.Path("benchmark/questions.json")
RESULTS_DIR = pathlib.Path("benchmark/results")
console = Console()


def _color(score: float) -> str:
    if score >= 0.75:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def _fmt(score: float) -> str:
    color = _color(score)
    return f"[{color}]{score:.2f}[/{color}]"


def run_eval(
    collection: str = "docs",
    top_k: int = 4,
    save: bool = True,
    show_answers: bool = False,
):
    if not BENCHMARK_PATH.exists():
        console.print(f"[red]Benchmark file not found:[/red] {BENCHMARK_PATH}")
        raise typer.Exit(1)

    questions = json.loads(BENCHMARK_PATH.read_text())
    rag = AdvancedRAG(collection, debug=False)

    results = []
    summary = {
        "faithfulness": [],
        "answer_relevance": [],
        "context_precision": [],
        "ground_truth_similarity": [],
    }

    console.print(f"\n[bold]Running eval on {len(questions)} questions[/bold] — collection: [cyan]{collection}[/cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(questions))

        for q in questions:
            progress.update(task, description=f"[dim]{q['id']}[/dim] {q['question'][:60]}...")

            meta = rag._query_with_metadata(q["question"], top_k=top_k)
            answer = meta["answer"]
            context = meta["context"]

            f_score = score_faithfulness(answer, context)
            ar_score = score_answer_relevance(q["question"], answer)
            cp_score = score_context_precision(meta["total_retrieved"], meta["total_kept"])
            gt_score = score_ground_truth_similarity(answer, q["ground_truth"])

            result = {
                "id": q["id"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "answer": answer,
                "sub_queries": meta["sub_queries"],
                "total_retrieved": meta["total_retrieved"],
                "total_kept": meta["total_kept"],
                "scores": {
                    "faithfulness": f_score,
                    "answer_relevance": ar_score,
                    "context_precision": cp_score,
                    "ground_truth_similarity": gt_score,
                },
            }
            results.append(result)

            summary["faithfulness"].append(f_score["score"])
            summary["answer_relevance"].append(ar_score["score"])
            summary["context_precision"].append(cp_score["score"])
            summary["ground_truth_similarity"].append(gt_score["score"])

            progress.advance(task)

    # --- Results table ---
    table = Table(title="Eval Results", show_lines=True)
    table.add_column("ID", style="dim", width=5)
    table.add_column("Question", width=38)
    table.add_column("Faith.", justify="center", width=7)
    table.add_column("Relevance", justify="center", width=9)
    table.add_column("Precision", justify="center", width=9)
    table.add_column("GT Sim.", justify="center", width=8)

    for r in results:
        s = r["scores"]
        table.add_row(
            r["id"],
            r["question"][:60] + ("..." if len(r["question"]) > 60 else ""),
            _fmt(s["faithfulness"]["score"]),
            _fmt(s["answer_relevance"]["score"]),
            _fmt(s["context_precision"]["score"]),
            _fmt(s["ground_truth_similarity"]["score"]),
        )

    # Averages row
    avgs = {k: round(sum(v) / len(v), 4) for k, v in summary.items()}
    table.add_row(
        "[bold]AVG[/bold]",
        "",
        f"[bold]{_fmt(avgs['faithfulness'])}[/bold]",
        f"[bold]{_fmt(avgs['answer_relevance'])}[/bold]",
        f"[bold]{_fmt(avgs['context_precision'])}[/bold]",
        f"[bold]{_fmt(avgs['ground_truth_similarity'])}[/bold]",
    )

    console.print(table)

    # --- Summary panel ---
    summary_lines = "\n".join([
        f"  Faithfulness:            {_fmt(avgs['faithfulness'])}",
        f"  Answer relevance:        {_fmt(avgs['answer_relevance'])}",
        f"  Context precision:       {_fmt(avgs['context_precision'])}",
        f"  Ground truth similarity: {_fmt(avgs['ground_truth_similarity'])}",
    ])
    console.print(Panel(summary_lines, title="[bold]Summary[/bold]", border_style="blue"))

    # --- Show answers if requested ---
    if show_answers:
        for r in results:
            console.rule(f"[dim]{r['id']}[/dim] {r['question']}")
            console.print(f"[green]Answer:[/green] {r['answer']}\n")
            console.print(f"[dim]Sub-queries: {r['sub_queries']}[/dim]")
            console.print(f"[dim]Faithfulness reason: {r['scores']['faithfulness']['reason']}[/dim]\n")

    # --- Save to JSON ---
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"eval_{timestamp}.json"
        payload = {
            "timestamp": timestamp,
            "collection": collection,
            "top_k": top_k,
            "num_questions": len(questions),
            "averages": avgs,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        console.print(f"\n[dim]Results saved to {out_path}[/dim]")

    return avgs