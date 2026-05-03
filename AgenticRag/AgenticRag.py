"""
KnowledgeHub — TRUE Agentic RAG
Pipeline : Query Decomposition → Retrieval → Self-Reflection → Iterative Search → Answer
C'est un vrai agent qui raisonne, évalue et re-cherche si nécessaire.
"""

import os
import uuid
import json
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
MAX_ITERATIONS  = 3  # nombre max de re-recherches si réponse insuffisante

# Dossier contenant tous les PDFs
PDF_FOLDER = r"C:\Users\chbel bh\Desktop\ps2\AgenticRag\docs"

console = Console()

# ── Singletons ────────────────────────────────────────────────────────────────

_embeddings = None
_llm        = None
_qdrant     = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        console.print("[yellow]⏳ Chargement embeddings...[/yellow]")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENROUTER_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=2000,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "KnowledgeHub"
            }
        )
    return _llm


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant

# ── Ingestion ─────────────────────────────────────────────────────────────────

def index_pdf(pdf_path: str, org_id: str) -> int:
    if not os.path.exists(pdf_path):
        console.print(f"[red]❌ Fichier introuvable : {pdf_path}[/red]")
        return 0

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                pages.append({"text": text, "page": i + 1})

    console.print(f"[green]✅ {len(pages)} pages lues[/green]")

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
    console.print("[yellow]⏳ Génération des embeddings...[/yellow]")

    emb     = get_embeddings()
    vectors = emb.embed_documents([c["text"] for c in chunks])

    client     = get_qdrant()
    collection = f"org_{org_id}"

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

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

def retrieve(query: str, org_id: str, top_k: int = TOP_K) -> list[dict]:
    """Recherche sémantique dans Qdrant pour une query donnée."""
    emb          = get_embeddings()
    query_vector = emb.embed_query(query)
    client       = get_qdrant()
    collection   = f"org_{org_id}"

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        console.print(f"[red]❌ Aucun document indexé pour '{org_id}'.[/red]")
        return []

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        score_threshold=0.25
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

# ── AGENT STEPS ───────────────────────────────────────────────────────────────

def step_decompose_query(question: str) -> list[str]:
    """
    ÉTAPE 1 — Query Decomposition.
    L'agent décompose une question complexe en sous-questions simples.
    Exemple : "Quels sont les objectifs et les types de bruit du TP2 ?"
    → ["Quels sont les objectifs du TP2 ?", "Quels sont les types de bruit ?"]
    Si la question est simple → retourne juste la question originale.
    """
    console.print("[cyan]🧠 Étape 1 — Décomposition de la question...[/cyan]")

    prompt = f"""Analyse cette question et décompose-la en sous-questions simples si nécessaire.
Si la question est déjà simple, retourne-la telle quelle.

Question : {question}

Réponds UNIQUEMENT avec un JSON valide, sans texte avant ou après :
{{"queries": ["sous-question 1", "sous-question 2"]}}

Si la question est simple :
{{"queries": ["{question}"]}}"""

    llm      = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        # Nettoie la réponse et parse le JSON
        content = response.content.strip()
        # Cherche le JSON dans la réponse
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start != -1 and end != 0:
            data    = json.loads(content[start:end])
            queries = data.get("queries", [question])
            console.print(f"[green]   → {len(queries)} sous-question(s) identifiée(s)[/green]")
            for q in queries:
                console.print(f"   • [dim]{q}[/dim]")
            return queries
    except Exception:
        pass

    return [question]


def step_retrieve_all(queries: list[str], org_id: str) -> list[dict]:
    """
    ÉTAPE 2 — Multi-Query Retrieval.
    Cherche les chunks pour CHAQUE sous-question.
    Fusionne et déduplique les résultats.
    """
    console.print("[cyan]🔍 Étape 2 — Recherche multi-queries...[/cyan]")

    all_chunks = []
    seen_texts = set()

    for query in queries:
        chunks = retrieve(query, org_id)
        for chunk in chunks:
            # Déduplique par contenu
            key = chunk["text"][:100]
            if key not in seen_texts:
                seen_texts.add(key)
                all_chunks.append(chunk)

    # Trie par score décroissant
    all_chunks.sort(key=lambda x: x["score"], reverse=True)

    console.print(f"[green]   → {len(all_chunks)} chunks uniques trouvés[/green]")
    for c in all_chunks[:6]:
        console.print(
            f"   • [cyan]{c['source']}[/cyan] "
            f"p.{c['page']} — score: [bold]{c['score']}[/bold]"
        )

    return all_chunks


def step_self_reflect(question: str, chunks: list[dict]) -> dict:
    """
    ÉTAPE 3 — Self-Reflection.
    L'agent évalue si les chunks trouvés sont suffisants pour répondre.
    Si non → identifie ce qui manque pour re-chercher.
    C'est ce qui rend le système vraiment AGENTIC.
    """
    console.print("[cyan]🤔 Étape 3 — Auto-évaluation...[/cyan]")

    context = "\n\n".join([
        f"[{c['source']}, p.{c['page']}] {c['text'][:200]}..."
        for c in chunks[:5]
    ])

    prompt = f"""Tu es un agent RAG. Évalue si le contexte suivant est suffisant pour répondre à la question.

Question : {question}

Contexte disponible :
{context}

Réponds UNIQUEMENT avec un JSON valide :
{{
  "sufficient": true/false,
  "confidence": 0.0-1.0,
  "missing": "ce qui manque (vide si sufficient=true)",
  "additional_query": "query supplémentaire si needed (vide si sufficient=true)"
}}"""

    llm      = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = response.content.strip()
        start   = content.find("{")
        end     = content.rfind("}") + 1
        if start != -1 and end != 0:
            result = json.loads(content[start:end])
            sufficient = result.get("sufficient", True)
            confidence = result.get("confidence", 0.5)

            if sufficient:
                console.print(f"[green]   → Contexte suffisant (confiance: {confidence:.0%})[/green]")
            else:
                missing = result.get("missing", "")
                console.print(f"[yellow]   → Contexte insuffisant — manque: {missing}[/yellow]")

            return result
    except Exception:
        pass

    return {"sufficient": True, "confidence": 0.5, "missing": "", "additional_query": ""}


def step_detect_contradictions(chunks: list[dict]) -> list[str]:
    """
    ÉTAPE 4 — Contradiction Detection.
    Détecte les informations contradictoires entre les chunks.
    """
    negations    = ["ne pas", "n'est pas", "interdit", "impossible", "jamais", "aucun"]
    affirmations = ["est", "peut", "autorisé", "possible", "toujours", "obligatoire"]

    warnings = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            ti = chunks[i]["text"].lower()
            tj = chunks[j]["text"].lower()

            if (any(w in ti for w in negations) and any(w in tj for w in affirmations)) or \
               (any(w in tj for w in negations) and any(w in ti for w in affirmations)):
                warnings.append(
                    f"⚠️  Contradiction : {chunks[i]['source']} p.{chunks[i]['page']} "
                    f"↔ {chunks[j]['source']} p.{chunks[j]['page']}"
                )
    return warnings


def step_generate(question: str, chunks: list[dict], contradictions: list[str]) -> str:
    """
    ÉTAPE 5 — Answer Generation.
    Génère la réponse finale avec Llama 3.1 70B.
    """
    console.print("[cyan]✍️  Étape 5 — Génération de la réponse...[/cyan]")

    context = "\n\n---\n\n".join([
        f"[Source {i+1} — {c['source']}, Page {c['page']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    contradiction_warning = ""
    if contradictions:
        contradiction_warning = f"\n⚠️ CONTRADICTIONS DÉTECTÉES :\n" + "\n".join(contradictions) + "\n"

    system = """Tu es un assistant expert en analyse documentaire.

RÈGLES :
1. Réponds UNIQUEMENT en te basant sur les sources fournies
2. Si l'information est absente → dis-le clairement
3. Cite toujours tes sources [Source X, Page Y]
4. Si des contradictions existent → signale-les dans ta réponse
5. Réponds en français si la question est en français
6. Sois précis, structuré et professionnel"""

    user = f"""DOCUMENTS :
{context}
{contradiction_warning}
QUESTION : {question}

Réponds de façon structurée en citant les sources [Source X, Page Y]."""

    llm      = get_llm()
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user)
    ])
    return response.content

# ── MAIN AGENTIC PIPELINE ─────────────────────────────────────────────────────

def agentic_rag(question: str, org_id: str) -> None:
    """
    Pipeline Agentic RAG COMPLET :

    1. Query Decomposition  → décompose la question
    2. Multi-Query Retrieval → cherche pour chaque sous-question
    3. Self-Reflection       → évalue si c'est suffisant
    4. Iterative Retrieval   → re-cherche si nécessaire (max 3x)
    5. Contradiction Detection → détecte les conflits
    6. Answer Generation    → génère la réponse finale
    """
    console.rule()
    console.print(Panel(
        f"[bold blue]{question}[/bold blue]",
        title="❓ Question",
        border_style="blue"
    ))

    # ÉTAPE 1 — Décomposition
    queries = step_decompose_query(question)

    # ÉTAPE 2 — Retrieval multi-queries
    all_chunks = step_retrieve_all(queries, org_id)

    if not all_chunks:
        console.print("[red]Aucun document pertinent trouvé.[/red]")
        return

    # ÉTAPE 3 — Self-reflection + Iterative retrieval
    iteration = 0
    while iteration < MAX_ITERATIONS:
        reflection = step_self_reflect(question, all_chunks)

        # Si suffisant → on arrête
        if reflection.get("sufficient", True):
            break

        # Si insuffisant → on re-cherche avec la query additionnelle
        additional_query = reflection.get("additional_query", "")
        if not additional_query:
            break

        iteration += 1
        console.print(f"[yellow]🔄 Itération {iteration} — Re-recherche : '{additional_query}'[/yellow]")

        # Cherche des chunks supplémentaires
        extra_chunks = retrieve(additional_query, org_id, top_k=4)
        seen = {c["text"][:100] for c in all_chunks}
        for chunk in extra_chunks:
            if chunk["text"][:100] not in seen:
                all_chunks.append(chunk)
                seen.add(chunk["text"][:100])

        console.print(f"[green]   → {len(extra_chunks)} chunks supplémentaires trouvés[/green]")

    console.print(f"[dim]   Itérations effectuées : {iteration}[/dim]")

    # ÉTAPE 4 — Détection contradictions
    console.print("[cyan]🔎 Étape 4 — Détection de contradictions...[/cyan]")
    contradictions = step_detect_contradictions(all_chunks)
    if contradictions:
        for w in contradictions:
            console.print(f"[red]{w}[/red]")
    else:
        console.print("[green]   → Aucune contradiction détectée[/green]")

    # ÉTAPE 5 — Génération
    answer = step_generate(question, all_chunks[:8], contradictions)

    # ÉTAPE 6 — Affichage
    confidence = round(sum(c["score"] for c in all_chunks) / len(all_chunks) * 100, 1)
    sources    = list(set(f"{c['source']} p.{c['page']}" for c in all_chunks))

    console.print(Panel(answer, title="💬 Réponse", border_style="green"))
    console.print(f"[dim]Sources : {' | '.join(sources[:5])}[/dim]")
    console.print(f"[dim]Confiance : {confidence}% | "
                  f"Chunks : {len(all_chunks)} | "
                  f"Itérations : {iteration} | "
                  f"Contradictions : {len(contradictions)}[/dim]")

# ── Interface Console ─────────────────────────────────────────────────────────

def main():
    console.print(Panel.fit(
        "[bold blue]KnowledgeHub — TRUE Agentic RAG[/bold blue]\n"
        "[dim]Query Decomposition → Multi-Retrieval → Self-Reflection → Iterative Search[/dim]\n"
        "[dim]Llama 3.1 70B · paraphrase-multilingual-mpnet · Qdrant[/dim]",
        border_style="blue"
    ))

    org_id = Prompt.ask("Organisation ID", default="test")

    # Indexe automatiquement tous les PDFs du dossier
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        console.print(f"[red]❌ Dossier créé mais vide : {PDF_FOLDER}[/red]")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        console.print(f"[red]❌ Aucun PDF dans {PDF_FOLDER}[/red]")
        return

    console.print(f"\n[yellow]📚 {len(pdf_files)} PDF(s) — indexation automatique...[/yellow]")
    for pdf_file in pdf_files:
        console.print(f"\n[cyan]📄 {pdf_file}[/cyan]")
        index_pdf(os.path.join(PDF_FOLDER, pdf_file), org_id)

    console.print("\n[bold green]✅ Tous les documents indexés ![/bold green]")
    console.print("[dim]Tape 'exit' pour quitter[/dim]\n")

    while True:
        question = Prompt.ask("[bold blue]Question[/bold blue]")

        if question.lower() == "exit":
            console.print("[yellow]Au revoir ![/yellow]")
            break

        agentic_rag(question, org_id)


if __name__ == "__main__":
    main()
