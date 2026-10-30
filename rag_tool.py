"""
EP2 - IT Assist Multi-Agent Orchestration
Tool RAG: Envuelve el índice FAISS de EP1 como una tool de CrewAI.

Arquitectura:
    - Embeddings: Ollama (nomic-embed-text) → 768 dimensiones
    - Vector Store: FAISS (índice pre-construido en EP1)
    - Recuperación: top-k con similitud coseno

Uso:
    from rag_tool import ITKnowledgeBaseTool
    tool = ITKnowledgeBaseTool()
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
import requests

# ─── Configuración ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "it_assist.index"
CHUNKS_PATH = VECTORSTORE_DIR / "chunks.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
SIM_THRESHOLD = float(os.getenv("RAG_SIM_THRESHOLD", "0.45"))
RAG_TIMEOUT_SECONDS = int(os.getenv("RAG_TIMEOUT_SECONDS", "15"))


# ─── Motor RAG (interno) ─────────────────────────────────────────────────────

class _ITRAGEngine:
    """Motor RAG que carga FAISS + chunks y expone search()."""

    def __init__(self):
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def _embed(self, text: str) -> np.ndarray:
        """Obtiene embedding vía Ollama REST API."""
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=RAG_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["embedding"], dtype="float32")
        return vec

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Busca los top_k chunks más relevantes."""
        query_vec = self._embed(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            sim = float(dist)
            if sim < SIM_THRESHOLD:
                continue
            chunk = self.chunks[idx]
            results.append({
                "score": round(sim, 4),
                "archivo": chunk.get("archivo", "desconocido"),
                "pagina": chunk.get("pagina", 1),
                "tipo_fuente": chunk.get("tipo_fuente", "interna"),
                "texto": chunk.get("texto", ""),
            })
        return results


# Instancia singleton (se carga una sola vez)
_engine: _ITRAGEngine | None = None


def _get_engine() -> _ITRAGEngine:
    global _engine
    if _engine is None:
        _engine = _ITRAGEngine()
    return _engine


# ─── Formateo de resultados ──────────────────────────────────────────────────

def _format_results(results: list[dict], query: str) -> str:
    """Formatea los chunks recuperados para el LLM."""
    if not results:
        return (
            f"Consulta: {query}\n\n"
            "No se encontraron fragmentos relevantes en la base de conocimiento "
            "de IT. La pregunta podría estar fuera del alcance de las políticas "
            "internas y manuales corporativos (contraseñas, VPN, correo, software, soporte TI)."
        )

    parts = [f"Consulta: {query}\n"]
    for i, r in enumerate(results, 1):
        tag = "INT" if r["tipo_fuente"] == "interna" else "EXT"
        parts.append(
            f"[{i}] (Fuente: {r['archivo']} | Pág {r['pagina']} | {tag} | sim={r['score']})\n"
            f"{r['texto']}\n"
        )
    parts.append(
        "\n---\n"
        "Instrucciones: Usa SOLO los fragmentos arriba para responder. "
        "Cita la fuente. Si la información no está, indícalo."
    )
    return "\n".join(parts)


# ─── Tool para CrewAI ────────────────────────────────────────────────────────

def buscar_base_conocimiento(consulta: str) -> str:
    """
    Busca en la base de conocimiento de IT (políticas internas, manuales,
    procedimientos de soporte) y retorna los fragmentos más relevantes
    con su fuente, página y tipo (interna/externa).

    Usa esta tool cuando necesites información específica sobre:
    - Contraseñas y recuperación de cuenta
    - Conexión VPN corporativa
    - Correo electrónico y Outlook
    - Instalación de software autorizado
    - Procedimiento de tickets de soporte TI
    - Recomendaciones de seguridad (CISA, Microsoft)

    Argumentos:
        consulta: La pregunta o tema a buscar en la base de conocimiento.
                  Ejemplo: "Como recupero mi contrasena olvidada"

    Retorno:
        Texto formateado con los fragmentos relevantes y sus fuentes.
    """
    engine = _get_engine()
    results = engine.search(consulta)
    return _format_results(results, consulta)


# ─── Test manual ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: RAG Tool - IT Assist")
    print("=" * 60)
    test_queries = [
        "Olvide mi contrasena corporativa, que debo hacer?",
        "Como me conecto a la VPN desde mi casa?",
        "Puedo instalar Spotify en el computador de la empresa?",
    ]
    for q in test_queries:
        print(f"\n{'─' * 50}")
        print(f"PREGUNTA: {q}")
        print(f"{'─' * 50}")
        result = buscar_base_conocimiento(q)
        print(result[:800])
        print("...")
