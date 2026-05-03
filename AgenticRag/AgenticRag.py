"""
KnowledgeHub — Agentic RAG
Un seul fichier, zéro dépendance inutile.
Stack : Qdrant + paraphrase-multilingual-mpnet + Llama 3.1 70B (OpenRouter)
"""

import os
import uuid
import pdfplumber
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import track
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
OPENROUTER_KEY  = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL       = "meta-llama/llama-3.1-70b-instruct"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 100
TOP_K           = 6
VECTOR_SIZE     = 768

# Dossier contenant tous les PDFs — indexés automatiquement au démarrage
PDF_FOLDER = r"C:\Users\chbel bh\Desktop\ps2\AgenticRag\docs"

console = Console()

# ── Singletons ────────────────────────────────────────────────────────────────

_embeddings = None
_llm        = None
_qdrant     = None


def get_embeddings():
    """
    Charge le modèle d'embeddings une seule fois en mémoire.
    paraphrase-multilingual-mpnet = excellent pour le français.
    normalize_embeddings=True améliore la similarité cosine.
    """
    global _embeddings
    if _embeddings is None:
        console.print("[yellow]⏳ Chargement embeddings (première fois ~30s)...[/yellow]")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings


def get_llm():
    """
    Initialise Llama 3.1 70B via OpenRouter.
    temperature=0.1 = réponses précises et cohérentes.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENROUTER_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=1500,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "KnowledgeHub"
            }
        )
    return _llm


def get_qdrant():
    """Connexion Qdrant réutilisée."""
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant

# ── Ingestion ─────────────────────────────────────────────────────────────────

def index_pdf(pdf_path: str, org_id: str) -> int:
    """
    Pipeline complet d'ingestion d'un PDF :
    1. Lit chaque page avec pdfplumber (gère bien les accents français)
    2. Découpe en chunks de 600 chars avec overlap de 100
    3. Génère les embeddings (vecteurs 768D)
    4. Stocke dans Qdrant dans la collection de l'organisation
    """
    if not os.path.exists(pdf_path):
        console.print(f"[red]❌ Fichier introuvable : {pdf_path}[/red]")
        return 0

    # ── Lecture PDF ───────────────────────────────────────
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                pages.append({"text": text, "page": i + 1})

    console.print(f"[green]✅ {len(pages)} pages lues[/green]")

    # ── Découpage en chunks ───────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ";", " "]
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append({
                "text":   split,
                "page":   page["page"],
                "source": os.path.basename(pdf_path)
            })

    console.print(f"[green]✅ {len(chunks)} chunks créés[/green]")

    # ── Embeddings ────────────────────────────────────────
    console.print("[yellow]⏳ Génération des embeddings...[/yellow]")
    emb     = get_embeddings()
    vectors = emb.embed_documents([c["text"] for c in chunks])

    # ── Qdrant : crée collection si besoin ────────────────
    client     = get_qdrant()
    collection = f"org_{org_id}"

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        console.print(f"[green]✅ Collection '{collection}' créée[/green]")

    # ── Stockage par batch de 100 ─────────────────────────
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={
                "text":   chunks[i]["text"],
                "page":   chunks[i]["page"],
                "source": chunks[i]["source"]
            }
        )
        for i in track(range(len(chunks)), description=f"Indexation {os.path.basename(pdf_path)}...")
    ]

    for i in range(0, len(points), 100):
        client.upsert(collection_name=collection, points=points[i:i + 100])

    console.print(f"[bold green]✅ {len(points)} chunks indexés[/bold green]")
    return len(points)

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str, org_id: str) -> list[dict]:
    """
    Recherche sémantique :
    1. Transforme la question en vecteur
    2. Cherche les TOP_K chunks les plus proches dans Qdrant
    3. Filtre les résultats avec score < 0.3 (peu pertinents)
    """
    emb          = get_embeddings()
    query_vector = emb.embed_query(question)

    client     = get_qdrant()
    collection = f"org_{org_id}"

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        console.print(f"[red]❌ Aucun document indexé pour '{org_id}'.[/red]")
        return []

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
        score_threshold=0.3
    ).points

    return [
        {
            "text":   r.payload["text"],
            "source": r.payload.get("source", "inconnu"),
            "page":   r.payload.get("page", 0),
            "score":  round(r.score, 3)
        }
        for r in results
    ]

# ── Contradiction Detection ───────────────────────────────────────────────────

def detect_contradictions(chunks: list[dict]) -> list[str]:
    """
    Détecte les contradictions potentielles entre les chunks.
    Cherche des patterns affirmatifs/négatifs sur le même sujet.
    """
    negations    = ["ne pas", "n'est pas", "interdit", "impossible", "jamais", "aucun"]
    affirmations = ["est", "peut", "autorisé", "possible", "toujours", "obligatoire"]

    warnings = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            ti = chunks[i]["text"].lower()
            tj = chunks[j]["text"].lower()

            neg_i = any(w in ti for w in negations)
            aff_j = any(w in tj for w in affirmations)
            neg_j = any(w in tj for w in negations)
            aff_i = any(w in ti for w in affirmations)

            if (neg_i and aff_j) or (neg_j and aff_i):
                warnings.append(
                    f"⚠️  Contradiction potentielle : "
                    f"{chunks[i]['source']} p.{chunks[i]['page']} "
                    f"↔ {chunks[j]['source']} p.{chunks[j]['page']}"
                )
    return warnings

# ── Generation ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant expert en analyse documentaire.

RÈGLES :
1. Réponds UNIQUEMENT en te basant sur les sources fournies
2. Si l'information est absente → dis-le clairement
3. Cite toujours tes sources [Source X, Page Y]
4. Réponds en français si la question est en français
5. Sois précis, structuré et professionnel"""


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Génère la réponse finale avec Llama 3.1 70B.
    Construit un prompt avec le contexte des chunks récupérés.
    """
    context = "\n\n---\n\n".join([
        f"[Source {i+1} — {c['source']}, Page {c['page']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    user_prompt = f"""DOCUMENTS :
{context}

QUESTION : {question}

Réponds en citant les sources [Source X, Page Y]."""

    llm      = get_llm()
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])
    return response.content

# ── Main Pipeline ─────────────────────────────────────────────────────────────

def agentic_rag(question: str, org_id: str) -> None:
    """
    Pipeline Agentic RAG complet :
    RETRIEVE → ANALYZE → GENERATE → DISPLAY
    """
    console.rule()
    console.print(f"[bold blue]❓ {question}[/bold blue]")

    # ÉTAPE 1 — Retrieve
    console.print("[yellow]🔍 Recherche sémantique...[/yellow]")
    chunks = retrieve(question, org_id)

    if not chunks:
        return

    for c in chunks:
        console.print(
            f"   • [cyan]{c['source']}[/cyan] "
            f"p.{c['page']} — score: [bold]{c['score']}[/bold]"
        )

    # ÉTAPE 2 — Analyze contradictions
    warnings = detect_contradictions(chunks)
    for w in warnings:
        console.print(f"[red]{w}[/red]")

    # ÉTAPE 3 — Generate
    console.print("[yellow]🤖 Génération réponse (Llama 3.1 70B)...[/yellow]")
    answer = generate_answer(question, chunks)

    # ÉTAPE 4 — Display
    confidence = round(sum(c["score"] for c in chunks) / len(chunks) * 100, 1)
    sources    = list(set(f"{c['source']} p.{c['page']}" for c in chunks))

    console.print(Panel(answer, title="💬 Réponse", border_style="green"))
    console.print(f"[dim]Sources : {' | '.join(sources)}[/dim]")
    console.print(f"[dim]Confiance : {confidence}% | "
                  f"Chunks : {len(chunks)} | "
                  f"Contradictions : {len(warnings)}[/dim]")

# ── Interface Console ─────────────────────────────────────────────────────────

def main():
    console.print(Panel.fit(
        "[bold blue]KnowledgeHub — Agentic RAG[/bold blue]\n"
        "[dim]Llama 3.1 70B · paraphrase-multilingual-mpnet · Qdrant[/dim]",
        border_style="blue"
    ))

    org_id = Prompt.ask("Organisation ID", default="test")

    # Indexe automatiquement tous les PDFs du dossier
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        console.print(f"[yellow]📁 Dossier créé : {PDF_FOLDER}[/yellow]")
        console.print("[red]❌ Aucun PDF trouvé — ajoute des PDFs dans le dossier docs/[/red]")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        console.print(f"[red]❌ Aucun PDF trouvé dans {PDF_FOLDER}[/red]")
        console.print("[yellow]Ajoute des PDFs dans le dossier docs/ et relance.[/yellow]")
        return

    console.print(f"\n[yellow]📚 {len(pdf_files)} PDF(s) trouvé(s) — indexation automatique...[/yellow]")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        console.print(f"\n[cyan]📄 {pdf_file}[/cyan]")
        index_pdf(pdf_path, org_id)

    console.print("\n[bold green]✅ Tous les documents sont indexés ![/bold green]")
    console.print("[dim]Tape 'exit' pour quitter[/dim]\n")

    while True:
        question = Prompt.ask("[bold blue]Question[/bold blue]")

        if question.lower() == "exit":
            console.print("[yellow]Au revoir ![/yellow]")
            break

        else:
            agentic_rag(question, org_id)


if __name__ == "__main__":
    main()