import typer
import chromadb
import pathlib
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pipeline import AdvancedRAG
from ingest import ingest_docs

app = typer.Typer(help="Advanced RAG CLI — decompose + HyDE + compression")
console = Console()

COLLECTION = "docs"
CHROMA_PATH = "./chroma_db"


@app.command()
def ingest(
    docs_dir: str = typer.Argument(..., help="Path to folder of markdown files"),
    collection: str = typer.Option(COLLECTION, "--collection", "-c", help="ChromaDB collection name"),
):
    """Index a folder of markdown files into ChromaDB."""
    path = pathlib.Path(docs_dir)
    if not path.exists():
        console.print(f"[red]Directory not found:[/red] {docs_dir}")
        raise typer.Exit(1)

    files = list(path.rglob("*.md"))
    if not files:
        console.print(f"[yellow]No markdown files found in {docs_dir}[/yellow]")
        raise typer.Exit(1)

    console.print(f"[bold]Ingesting {len(files)} file(s) into collection '[cyan]{collection}[/cyan]'...[/bold]")
    total_chunks = ingest_docs(docs_dir, collection)
    console.print(f"[green]Done.[/green] {total_chunks} chunks indexed.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question"),
    collection: str = typer.Option(COLLECTION, "--collection", "-c", help="ChromaDB collection name"),
    top_k: int = typer.Option(4, "--top-k", "-k", help="Chunks to retrieve per sub-query"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Print sub-queries and compressed chunks"),
):
    """Ask a question using the full RAG pipeline."""
    try:
        rag = AdvancedRAG(collection, debug=debug)
    except Exception as e:
        console.print(f"[red]Failed to load collection '{collection}':[/red] {e}")
        console.print("Run [bold]ingest[/bold] first.")
        raise typer.Exit(1)

    with console.status("[bold green]Thinking...[/bold green]"):
        answer = rag.query(question, top_k=top_k)

    console.print(Panel(answer, title="[bold]Answer[/bold]", border_style="green"))


@app.command()
def status(
    collection: str = typer.Option(COLLECTION, "--collection", "-c", help="ChromaDB collection name"),
):
    """Show collection info and model configuration."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_collection(collection)
        count = col.count()
    except Exception:
        console.print(f"[yellow]Collection '[cyan]{collection}[/cyan]' not found. Run ingest first.[/yellow]")
        raise typer.Exit(1)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Collection", f"[cyan]{collection}[/cyan]")
    table.add_row("Chunks", str(count))
    table.add_row("ChromaDB path", CHROMA_PATH)
    table.add_row("Embedder", "all-MiniLM-L6-v2")
    table.add_row("Decompose model", "qwen2.5-coder:3b")
    table.add_row("HyDE model", "llama3.2")
    table.add_row("Compress model", "qwen2.5-coder:3b")
    table.add_row("Synthesis model", "llama3.2")

    console.print(Panel(table, title="[bold]RAG Status[/bold]", border_style="blue"))


@app.command()
def clear(
    collection: str = typer.Option(COLLECTION, "--collection", "-c", help="ChromaDB collection name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Wipe a ChromaDB collection."""
    if not yes:
        confirm = typer.confirm(f"Delete all chunks in collection '{collection}'?")
        if not confirm:
            console.print("Aborted.")
            raise typer.Exit()

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection(collection)
        console.print(f"[green]Collection '[cyan]{collection}[/cyan]' cleared.[/green]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def eval(
    collection: str = typer.Option(COLLECTION, "--collection", "-c", help="ChromaDB collection name"),
    top_k: int = typer.Option(4, "--top-k", "-k", help="Chunks to retrieve per sub-query"),
    no_save: bool = typer.Option(False, "--no-save", help="Skip saving results to JSON"),
    show_answers: bool = typer.Option(False, "--show-answers", "-a", help="Print each answer after the table"),
):
    """Run the full eval benchmark and score the pipeline."""
    from eval import run_eval
    run_eval(
        collection=collection,
        top_k=top_k,
        save=not no_save,
        show_answers=show_answers,
    )


if __name__ == "__main__":
    app()