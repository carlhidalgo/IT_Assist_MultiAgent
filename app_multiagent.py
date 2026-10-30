"""
EP2 - IT Assist Multi-Agent | Aplicacion Streamlit
Tabs: Consultas | Trazabilidad | Testeo comparativo
"""

import json
import sys
import os
import time
import re
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv

load_dotenv()

from loguru import logger
from rag_tool import buscar_base_conocimiento, _get_engine

# ─── Logging config ───────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)
logger.add("pipeline.log", rotation="5 MB", level="DEBUG", format="{time} | {level} | {message}")

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EP2 - IT Assist Multi-Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; }
    h2 { color: #34495e; }
    .stTabs [data-baseweb="tab"] { font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _validate_response(response: str) -> dict:
    """Valida que la respuesta contenga campos clave."""
    text_lower = response.lower()
    checks = {
        "pasos": bool(re.search(r"\d+[\.\)]\s", response)) or "paso" in text_lower,
        "fuente": any(k in text_lower for k in ["fuente", "archivo", "pagina", "pág"]),
        "escalamiento": any(k in text_lower for k in ["ticket", "mesa de ayuda", "escal"]),
    }
    missing = [k for k, v in checks.items() if not v]
    return {"valid": len(missing) == 0, "checks": checks, "missing": missing}


def _get_fallback(consulta: str, missing: list) -> str:
    """Genera respuesta de fallback."""
    return (
        f"⚠️ **Atención**: La respuesta no incluyó: {', '.join(missing)}.\n\n"
        f"Para tu consulta *\"{consulta}\"*:\n\n"
        f"1. Revisa la base de conocimiento interna.\n"
        f"2. Si el problema persiste, abre un ticket en **Mesa de Ayuda**.\n"
        f"3. Contacta a soporte@empresa.cl si es crítico.\n"
    )


def _get_priority(response: str) -> tuple[str, str]:
    """Detecta la prioridad informada por el analista."""
    text = response.lower()
    if any(value in text for value in ["crítica", "critica", "p1"]):
        return "P1 - Crítica", "error"
    if any(value in text for value in ["alta", "p2"]):
        return "P2 - Alta", "warning"
    if any(value in text for value in ["media", "p3"]):
        return "P3 - Media", "info"
    return "No determinada", "info"


def _ollama_status() -> tuple[bool, str]:
    """Comprueba si Ollama responde en su endpoint local."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        response.raise_for_status()
        return True, "Ollama conectado"
    except requests.RequestException:
        return False, "Ollama no disponible"


MEMORY_FILE = Path(__file__).resolve().parent / "memoria_larga.json"

OLLAMA_MODELS = ["llama3.2:3b", "qwen3.8:27b"]
GEMINI_MODELS = ["gemini-2.5-flash"]
MODEL_OPTIONS = OLLAMA_MODELS + GEMINI_MODELS
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))


def _configure_llm(model: str) -> None:
    """Configura el proveedor y modelo usado por CrewAI."""
    provider = "gemini" if model.startswith("gemini-") else "ollama"
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model


def _llm_config(temperature: float = 0.3):
    """Construye la configuración de CrewAI para Ollama o Gemini."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", OLLAMA_MODELS[0])
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Falta GEMINI_API_KEY. Configúrala en el archivo .env "
                "antes de usar un modelo Gemini."
            )
        from crewai import LLM

        return LLM(
            model=f"gemini/{model}",
            api_key=api_key,
            temperature=temperature,
            max_tokens=LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT_SECONDS,
        )
    return {
        "model": model,
        "api_key": os.getenv("LLM_API_KEY", "ollama"),
        "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        "temperature": temperature,
        "max_tokens": LLM_MAX_TOKENS,
        "timeout": LLM_TIMEOUT_SECONDS,
    }


def _load_long_term_memory() -> list[dict]:
    """Carga las últimas interacciones persistidas."""
    if not MEMORY_FILE.exists():
        return []
    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"No se pudo cargar la memoria persistente: {exc}")
        return []


def _load_persistent_history() -> list[dict]:
    """Adapta la memoria persistente al formato mostrado en la app."""
    history = []
    for item in _load_long_term_memory():
        history.append({
            "consulta": item.get("consulta", ""),
            "pipeline": item.get("pipeline", "Desconocido"),
            "resultado": item.get("resultado", item.get("resumen", "")),
            "tiempo_s": item.get("tiempo_s", 0),
        })
    return history


def _load_test_history() -> list[dict]:
    """Carga el historial persistente de ejecuciones de testeo."""
    memories = _load_long_term_memory()
    return [
        item["test_run"]
        for item in memories
        if isinstance(item.get("test_run"), dict)
    ]


def _get_memory_context() -> str:
    """Formatea las últimas consultas para entregarlas como contexto."""
    memories = _load_long_term_memory()[-5:]
    if not memories:
        return "Sin interacciones previas almacenadas."
    lines = ["INTERACCIONES PREVIAS (memoria persistente):"]
    for item in memories:
        lines.append(
            f"- {item.get('consulta', '')[:120]}: "
            f"{item.get('resumen', '')[:180]}"
        )
    return "\n".join(lines)


def _save_long_term_memory(
    consulta: str,
    resumen: str,
    pipeline: str = "Desconocido",
    tiempo_s: float = 0,
) -> None:
    """Guarda una interacción y conserva como máximo 20 entradas."""
    from datetime import datetime

    memories = _load_long_term_memory()
    memories.append({
        "timestamp": datetime.now().isoformat(),
        "consulta": consulta,
        "resumen": resumen[:500],
        "resultado": resumen,
        "pipeline": pipeline,
        "tiempo_s": tiempo_s,
    })
    with MEMORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(memories[-20:], file, ensure_ascii=False, indent=2)


def _save_test_run(test_run: dict) -> None:
    """Persiste el resumen de una ejecución de batería."""
    memories = _load_long_term_memory()
    memories.append({
        "timestamp": test_run["timestamp"],
        "consulta": "[BATERÍA DE TESTEO]",
        "resumen": (
            f"{test_run['total']} preguntas, "
            f"{test_run['exitosas']} exitosas, "
            f"{test_run['errores']} errores"
        ),
        "test_run": test_run,
    })
    with MEMORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(memories[-20:], file, ensure_ascii=False, indent=2)


def run_multiagent_sequential(
    consulta: str,
    memory_context: str = "",
) -> dict:
    """Ejecuta pipeline secuencial con logging, métricas y validación."""
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool

    def _llm():
        return _llm_config()

    @tool("buscar_base_conocimiento_it")
    def buscar(consulta: str) -> str:
        """Busca en la base de conocimiento de IT.
        Argumentos:
            consulta: Pregunta o tema.
        """
        logger.debug(f"[RAG] Consulta: {consulta[:60]}")
        return buscar_base_conocimiento(consulta)

    ag_inv = Agent(
        role="Investigador IT",
        goal="Encontrar fragmentos relevantes con fuentes.",
        backstory="Especialista en busqueda semantica sobre base FAISS de IT.",
        tools=[buscar],
        llm=_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )
    ag_an = Agent(
        role="Analista IT",
        goal="Sintetizar: procedimiento, prioridad, pasos, fuentes.",
        backstory="Analista de soporte TI.",
        tools=[],
        llm=_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )
    ag_co = Agent(
        role="Comunicador IT",
        goal="Redactar respuesta final al usuario.",
        backstory="Traduce analisis tecnico a lenguaje accesible.",
        tools=[],
        llm=_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )

    t_inv = Task(
        description=(
            f"Consulta: {consulta}\n{memory_context}\n"
            "Busca en la base y retorna fragmentos con fuentes."
        ),
        expected_output="Fragmentos con fuentes.",
        agent=ag_inv,
    )
    t_an = Task(
        description=(
            f"{memory_context}\n"
            "Sintetiza hallazgos: procedimiento, prioridad, pasos, fuentes."
        ),
        expected_output="Analisis estructurado.",
        agent=ag_an,
        context=[t_inv],
    )
    t_co = Task(
        description=(
            f"{memory_context}\n"
            "Redacta respuesta final: saludo, pasos, fuente, escalamiento."
        ),
        expected_output="Respuesta final.",
        agent=ag_co,
        context=[t_inv, t_an],
    )

    # El callback de CrewAI se ejecuta al finalizar cada tarea. Medimos el
    # intervalo desde la tarea anterior para evitar depender de atributos
    # internos que no existen en Task.
    agent_timings: dict[str, float] = {}
    t_start = time.time()
    previous_task_end = t_start

    def on_task_end(output):
        nonlocal previous_task_end
        name = getattr(output, "agent", None) or "desconocido"
        now = time.time()
        agent_timings[name] = round(now - previous_task_end, 2)
        previous_task_end = now
        logger.info(f"✅ {name} → completado | Output: {len(str(output))} chars")

    logger.info(f"▶️  Pipeline iniciado | Consulta: {consulta[:50]}")

    crew = Crew(
        agents=[ag_inv, ag_an, ag_co],
        tasks=[t_inv, t_an, t_co],
        process=Process.sequential,
        verbose=False,
        task_callback=on_task_end,
    )
    result = crew.kickoff()
    raw_result = result.raw if result else "N/A"
    total_time = round(time.time() - t_start, 2)

    # Validación
    validation = _validate_response(raw_result)
    if not validation["valid"]:
        logger.warning(f"⚠️  Faltan campos: {validation['missing']}")

    logger.info(f"🏁 Pipeline completado en {total_time}s")

    return {
        "consulta": consulta,
        "resultado": raw_result,
        "validacion": validation,
        "latencia_total_s": total_time,
        "latencia_por_agente": agent_timings,
    }


def run_multiagent_hierarchical(
    consulta: str,
    memory_context: str = "",
) -> dict:
    """Ejecuta el pipeline jerárquico con un Manager de CrewAI."""
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool

    def _llm(temperature: float = 0.3):
        return _llm_config(temperature)

    @tool("buscar_base_conocimiento_it")
    def buscar(consulta: str) -> str:
        """Busca en la base de conocimiento de IT."""
        logger.debug(f"[RAG] Consulta: {consulta[:60]}")
        return buscar_base_conocimiento(consulta)

    @tool("clasificar_prioridad_ticket")
    def clasificar_prioridad(descripcion: str) -> str:
        """Clasifica la prioridad de un incidente IT."""
        texto = descripcion.lower()
        if any(k in texto for k in ["no puedo trabajar", "caido", "bloqueado", "emergencia", "critico"]):
            return "PRIORIDAD: CRITICA (P1) - SLA: 1 hora."
        if any(k in texto for k in ["vpn", "correo", "no funciona", "falla", "error"]):
            return "PRIORIDAD: ALTA (P2) - SLA: 4 horas."
        return "PRIORIDAD: MEDIA (P3) - SLA: 24 horas."

    ag_inv = Agent(
        role="Investigador IT",
        goal="Encontrar fragmentos relevantes con fuentes.",
        backstory="Especialista en búsqueda semántica sobre la base FAISS de IT.",
        tools=[buscar],
        llm=_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )
    ag_an = Agent(
        role="Analista IT",
        goal="Sintetizar procedimiento, prioridad, pasos y fuentes.",
        backstory="Analista de soporte TI que estructura la resolución técnica.",
        tools=[clasificar_prioridad],
        llm=_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )
    ag_co = Agent(
        role="Comunicador IT",
        goal="Redactar una respuesta final clara y accionable.",
        backstory="Traduce el análisis técnico a lenguaje accesible.",
        tools=[],
        llm=_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=1,
    )

    t_inv = Task(
        description=(
            f"Consulta: {consulta}\n{memory_context}\n"
            "Busca fragmentos relevantes con sus fuentes."
        ),
        expected_output="Fragmentos relevantes con fuentes citadas.",
        agent=ag_inv,
    )
    t_an = Task(
        description=(
            f"{memory_context}\n"
            "Analiza los hallazgos, clasifica prioridad, extrae pasos y cita fuentes."
        ),
        expected_output="Análisis estructurado con procedimiento, prioridad, pasos y fuentes.",
        agent=ag_an,
        context=[t_inv],
    )
    t_co = Task(
        description=(
            f"{memory_context}\n"
            "Redacta la respuesta final con saludo, pasos, fuente y escalamiento."
        ),
        expected_output="Respuesta final lista para enviar al usuario.",
        agent=ag_co,
        context=[t_inv, t_an],
    )

    agent_timings: dict[str, float] = {}
    t_start = time.time()

    def on_task_end(output):
        name = getattr(output, "agent", None) or "desconocido"
        agent_timings[name] = round(time.time() - t_start, 2)
        logger.info(f"✅ {name} → completado | Output: {len(str(output))} chars")

    logger.info(f"▶️ Pipeline jerárquico iniciado | Consulta: {consulta[:50]}")
    crew = Crew(
        agents=[ag_inv, ag_an, ag_co],
        tasks=[t_inv, t_an, t_co],
        process=Process.hierarchical,
        manager_llm=_llm(temperature=0.4),
        verbose=False,
        task_callback=on_task_end,
    )
    result = crew.kickoff()
    raw_result = result.raw if result else "N/A"
    validation = _validate_response(raw_result)
    total_time = round(time.time() - t_start, 2)
    if not validation["valid"]:
        logger.warning(f"⚠️ Faltan campos: {validation['missing']}")
    logger.info(f"🏁 Pipeline jerárquico completado en {total_time}s")

    return {
        "consulta": consulta,
        "resultado": raw_result,
        "validacion": validation,
        "latencia_total_s": total_time,
        "latencia_por_agente": agent_timings,
    }


def run_rag_search(consulta: str) -> list[dict]:
    """Ejecuta solo la busqueda RAG (para tab de trazabilidad)."""
    engine = _get_engine()
    return engine.search(consulta)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Configuración")

st.sidebar.markdown("**EP2 - IT Assist**")
st.sidebar.markdown("**Multi-Agent Orchestration**")
st.sidebar.markdown("---")

# Selector de modelo y proveedor
model_choice = st.sidebar.radio(
    "Modelo LLM",
    [
        "llama3.2:3b (Ollama, rápido ~10s)",
        "qwen3.8:27b (Ollama, calidad ~90s)",
        "gemini-2.5-flash (Google Gemini)",
    ],
    index=0,
    help="Gemini requiere GEMINI_API_KEY en .env y usa el adaptador nativo de CrewAI.",
)
_configure_llm(model_choice.split(" ")[0])

pipeline_choice = st.sidebar.radio(
    "Pipeline de orquestación",
    ["Secuencial", "Jerárquico"],
    index=0,
    help="Secuencial ejecuta un orden fijo; Jerárquico permite que un Manager coordine los agentes.",
)
memory_enabled = st.sidebar.checkbox(
    "Activar memoria persistente",
    value=True,
    help="Usa y guarda las últimas interacciones en memoria_larga.json.",
)
if st.sidebar.button("🗑️ Limpiar memoria persistente"):
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()
    st.sidebar.success("Memoria persistente eliminada.")
run_selected_pipeline = (
    run_multiagent_sequential
    if pipeline_choice == "Secuencial"
    else run_multiagent_hierarchical
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Stack:**")
provider_label = "Google Gemini" if os.environ["LLM_PROVIDER"] == "gemini" else "Ollama local"
st.sidebar.markdown(f"- LLM: {provider_label} ({os.environ['LLM_MODEL']})")
st.sidebar.markdown("- Embeddings: Ollama (nomic-embed-text)")
st.sidebar.markdown("- Vector Store: FAISS (EP1)")
st.sidebar.markdown(f"- Orquestación: CrewAI 3-agentes ({pipeline_choice})")
st.sidebar.markdown(
    f"- Memoria persistente: {'activa' if memory_enabled else 'desactivada'}"
)
ollama_ok, ollama_message = _ollama_status()
if ollama_ok:
    st.sidebar.success(f"🟢 {ollama_message}")
else:
    st.sidebar.warning(f"🟡 {ollama_message} (necesario para embeddings/RAG)")
if os.environ["LLM_PROVIDER"] == "gemini" and not os.getenv("GEMINI_API_KEY"):
    st.sidebar.error("🔴 Falta GEMINI_API_KEY en .env")

if "historial_consultas" not in st.session_state:
    st.session_state["historial_consultas"] = _load_persistent_history()

# ─── Header ──────────────────────────────────────────────────────────────────

st.title("🤖 IT Assist - Multi-Agent Orchestration (EP2)")
st.caption("Sistema de soporte TI")

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_consultas, tab_trazabilidad, tab_testeo = st.tabs([
    "💬 Consultas",
    "🔍 Trazabilidad (RAG)",
    "🧪 Testeo comparativo",
])

# ─── TAB 1: Consultas ────────────────────────────────────────────────────────

with tab_consultas:
    st.subheader("Consulta al equipo multi-agente")
    st.markdown(
        "Escribe tu pregunta de soporte TI. El pipeline "
        f"**{pipeline_choice}** procesará la consulta con Investigador, Analista y Comunicador."
    )

    if "consulta_input" not in st.session_state:
        st.session_state["consulta_input"] = ""

    st.markdown("**Preguntas de ejemplo:**")
    ejemplos = [
        "Olvidé mi contraseña corporativa",
        "La VPN me dice autenticación fallida",
        "¿Puedo instalar Spotify en el PC?",
        "¿Cómo creo un ticket de soporte?",
    ]
    cols = st.columns(2)
    for i, eg in enumerate(ejemplos):
        if cols[i % 2].button(eg, key=f"eg_{i}"):
            st.session_state["consulta_input"] = eg
            st.rerun()

    consulta = st.text_input(
        "Tu consulta:",
        placeholder="Ej: Olvidé mi contraseña corporativa, ¿qué debo hacer?",
        key="consulta_input",
    )

    if st.button("🆕 Nueva consulta"):
        st.session_state["consulta_input"] = ""
        st.session_state.pop("ultimo_resultado", None)
        st.rerun()

    if st.button("🚀 Ejecutar Pipeline Multi-Agent", type="primary"):
        if not consulta.strip():
            st.warning("⚠️ Escribe una consulta primero.")
        else:
            t0 = time.time()
            with st.spinner(f"⏳ Ejecutando pipeline {pipeline_choice.lower()}..."):
                try:
                    memory_context = _get_memory_context() if memory_enabled else ""
                    result = run_selected_pipeline(consulta, memory_context)
                    elapsed = round(time.time() - t0, 1)
                    st.session_state["ultimo_resultado"] = result
                    _save_long_term_memory(
                        consulta,
                        result["resultado"],
                        pipeline_choice,
                        elapsed,
                    )
                    history = st.session_state.setdefault("historial_consultas", [])
                    history.append({
                        "consulta": consulta,
                        "pipeline": pipeline_choice,
                        "resultado": result["resultado"],
                        "tiempo_s": elapsed,
                    })
                    st.session_state["historial_consultas"] = history[-20:]
                    st.success(f"✅ Pipeline completado en **{elapsed:.1f}s**")
                except Exception as e:
                    elapsed = time.time() - t0
                    st.error(f"❌ Error tras {elapsed:.1f}s: {e}")

    if "ultimo_resultado" in st.session_state:
        st.markdown("---")
        st.subheader("Resultado del Comunicador (respuesta final)")
        st.markdown(st.session_state["ultimo_resultado"]["resultado"])
        priority, priority_style = _get_priority(
            st.session_state["ultimo_resultado"]["resultado"]
        )
        getattr(st, priority_style)(
            f"**Prioridad detectada:** {priority}"
        )

        # Validación
        v = st.session_state["ultimo_resultado"].get("validacion", {})
        if v:
            st.markdown("---")
            if v.get("valid"):
                st.success("✅ **Validación OK**: respuesta completa (pasos + fuente + escalamiento)")
            else:
                st.warning(f"⚠️ **Validación**: faltan campos → {', '.join(v.get('missing', []))}")
                with st.expander("📋 Fallback sugerido"):
                    st.markdown(_get_fallback(st.session_state["ultimo_resultado"]["consulta"], v.get("missing", [])))

        # Métricas por agente
        lat = st.session_state["ultimo_resultado"].get("latencia_por_agente", {})
        total = st.session_state["ultimo_resultado"].get("latencia_total_s", 0)
        if lat:
            st.markdown("---")
            st.markdown("**⏱️ Latencia por agente:**")
            for agente, tiempo in lat.items():
                st.text(f"   • {agente}: {tiempo}s")
            st.text(f"   • **Total**: {total}s")

    st.markdown("---")
    st.subheader("Historial de consultas")
    st.caption(
        "El historial se conserva en memoria_larga.json y se recupera al recargar. "
        "El contexto solo se envía al agente si activas la memoria persistente."
    )
    history = st.session_state.get("historial_consultas", [])
    if not history:
        st.info("Aún no hay consultas ejecutadas.")
    else:
        for i, item in enumerate(reversed(history), 1):
            with st.expander(
                f"{i}. [{item['pipeline']}] {item['consulta'][:70]} ({item['tiempo_s']}s)"
            ):
                st.markdown(item["resultado"])
        history_df = pd.DataFrame(history)
        st.download_button(
            "📥 Descargar historial de consultas (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="historial_consultas.csv",
            mime="text/csv",
        )
        if st.button("🗑️ Limpiar historial"):
            st.session_state["historial_consultas"] = []
            if MEMORY_FILE.exists():
                MEMORY_FILE.unlink()
            st.rerun()

# ─── TAB 2: Trazabilidad ─────────────────────────────────────────────────────

with tab_trazabilidad:
    st.subheader("Trazabilidad RAG - Búsqueda en Base Vectorial")
    st.markdown(
        "Muestra los chunks recuperados de FAISS con sus metadatos "
        "(fuente, página, tipo, score). Esto permite auditar QUÉ "
        "información usó el sistema para responder."
    )

    q_trace = st.text_input(
        "Consulta para trazar:",
        placeholder="Ej: recuperación de contraseña",
        key="trace_input",
    )

    if st.button("🔍 Buscar y Trazar"):
        if not q_trace.strip():
            st.warning("⚠️ Escribe una consulta.")
        else:
            with st.spinner("Buscando en FAISS..."):
                try:
                    chunks = run_rag_search(q_trace)
                    st.session_state["trace_result"] = chunks
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    if "trace_result" in st.session_state:
        chunks = st.session_state["trace_result"]
        if not chunks:
            st.info("No se encontraron fragmentos por encima del umbral.")
        else:
            st.markdown(f"**{len(chunks)} fragmentos recuperados:**")
            for i, c in enumerate(chunks, 1):
                tipo = "📄 INT" if c["tipo_fuente"] == "interna" else "🌐 EXT"
                with st.expander(
                    f"[{i}] {c['archivo']} | Pág {c['pagina']} | {tipo} | sim={c['score']}"
                ):
                    st.text(c["texto"])

# ─── TAB 3: Testeo comparativo ───────────────────────────────────────────────

with tab_testeo:
    st.subheader("Batería de pruebas comparativas")
    st.markdown(
        "Ejecuta una selección de preguntas con uno o ambos pipelines y modelos. "
        "El cuadro final permite comparar rendimiento, éxito y tiempos."
    )

    test_pipeline_choice = st.multiselect(
        "Pipelines a probar",
        ["Secuencial", "Jerárquico"],
        default=[pipeline_choice],
        help="Puedes seleccionar uno o ambos pipelines.",
    )
    test_model_choice = st.multiselect(
        "Modelos a probar",
        MODEL_OPTIONS,
        default=MODEL_OPTIONS,
        help=(
            "Por defecto se seleccionan todos para crear un cuadro comparativo. "
            "Gemini requiere GEMINI_API_KEY."
        ),
    )
    if not test_pipeline_choice:
        st.warning("Selecciona al menos un pipeline.")
    if not test_model_choice:
        st.warning("Selecciona al menos un modelo.")

    csv_path = Path(__file__).resolve().parent / "preguntas_prueba.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        preguntas = df["pregunta"].tolist() if "pregunta" in df.columns else df.iloc[:, 0].tolist()
    else:
        preguntas = [
            "Olvidé mi contraseña corporativa",
            "No me conecta la VPN",
            "¿Puedo instalar Spotify?",
            "¿Cómo creo un ticket de soporte?",
            "El correo no carga",
            "¿Qué antivirus puedo usar?",
        ]

    question_count = st.selectbox(
        "Cantidad de preguntas",
        [1, 3, 6, 12],
        index=3,
        help="Selecciona cuántas preguntas se ejecutarán en cada combinación.",
    )
    if len(preguntas) < question_count:
        st.warning(
            f"Solo hay {len(preguntas)} preguntas disponibles; "
            f"se ejecutarán esas {len(preguntas)}."
        )
    preguntas = preguntas[:question_count]
    st.caption(
        f"Se ejecutarán {len(preguntas)} pregunta(s) por cada combinación "
        f"de pipeline y modelo."
    )

    if st.button(
        "▶️ Ejecutar pruebas seleccionadas",
        type="primary",
        disabled=not test_pipeline_choice or not test_model_choice,
    ):
        st.session_state.pop("test_results", None)
        results = []
        test_start = time.time()
        total_cases = len(preguntas) * len(test_pipeline_choice) * len(test_model_choice)
        completed_cases = 0
        progress = st.progress(0, text="Iniciando...")
        for test_model in test_model_choice:
            _configure_llm(test_model)
            for test_pipeline in test_pipeline_choice:
                test_runner = (
                    run_multiagent_sequential
                    if test_pipeline == "Secuencial"
                    else run_multiagent_hierarchical
                )
                for p in preguntas:
                    completed_cases += 1
                    progress.progress(
                        completed_cases / total_cases,
                        text=(
                            f"({completed_cases}/{total_cases}) "
                            f"{test_pipeline} + {test_model} | {p[:40]}..."
                        ),
                    )
                    try:
                        t0 = time.time()
                        memory_context = _get_memory_context() if memory_enabled else ""
                        res = test_runner(p, memory_context)
                        elapsed = round(time.time() - t0, 1)
                        _save_long_term_memory(p, res["resultado"], test_pipeline, elapsed)
                        results.append({
                            "pregunta": p,
                            "pipeline": test_pipeline,
                            "modelo": test_model,
                            "respuesta": res["resultado"][:200],
                            "tiempo_s": elapsed,
                            "ok": True,
                        })
                    except Exception as e:
                        results.append({
                            "pregunta": p,
                            "pipeline": test_pipeline,
                            "modelo": test_model,
                            "respuesta": f"ERROR: {e}",
                            "tiempo_s": 0,
                            "ok": False,
                        })
        _configure_llm(model_choice.split(" ")[0])
        progress.empty()
        st.session_state["test_results"] = results
        test_run = {
            "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            "pipeline": ", ".join(test_pipeline_choice),
            "modelo": ", ".join(test_model_choice),
            "total": len(results),
            "exitosas": sum(1 for item in results if item["ok"]),
            "errores": sum(1 for item in results if not item["ok"]),
            "tiempo_total_s": round(time.time() - test_start, 1),
            "resultados": results,
        }
        st.session_state["test_history"] = (
            st.session_state.get("test_history", []) + [test_run]
        )[-10:]
        _save_test_run(test_run)
        st.success(f"✅ Batería completada: {len(results)} preguntas.")

    test_history = st.session_state.get("test_history")
    if test_history is None:
        test_history = _load_test_history()
        st.session_state["test_history"] = test_history

    current_results = st.session_state.get("test_results", [])
    history_labels = [
        (
            f"{index}. {run['timestamp']} · {run.get('pipeline', 'N/D')} · "
            f"{run.get('modelo', 'modelo no registrado')}"
        )
        for index, run in enumerate(reversed(test_history), 1)
    ]
    result_options = (
        ["Ejecución actual"] + history_labels
        if current_results
        else history_labels
    )
    selected_result = None
    if result_options:
        selected_label = st.selectbox(
            "Resultados para mostrar en el cuadro comparativo",
            result_options,
            help="Selecciona la ejecución actual o una batería guardada en el historial.",
        )
        if selected_label == "Ejecución actual":
            selected_result = current_results
        else:
            selected_index = history_labels.index(selected_label)
            selected_result = list(reversed(test_history))[selected_index].get(
                "resultados", []
            )

    if selected_result:
        st.markdown("---")
        st.subheader("Comparación por pipeline y modelo")
        df_results = pd.DataFrame(selected_result)
        comparison = (
            df_results.groupby(["pipeline", "modelo"], as_index=False)
            .agg(
                preguntas=("pregunta", "count"),
                exitosas=("ok", "sum"),
                errores=("ok", lambda values: (~values).sum()),
                tiempo_promedio_s=("tiempo_s", "mean"),
                tiempo_total_s=("tiempo_s", "sum"),
            )
        )
        comparison["tasa_exito"] = (
            comparison["exitosas"] / comparison["preguntas"] * 100
        ).round(1)
        comparison[["tiempo_promedio_s", "tiempo_total_s"]] = comparison[
            ["tiempo_promedio_s", "tiempo_total_s"]
        ].round(1)
        comparison = comparison[
            [
                "pipeline",
                "modelo",
                "preguntas",
                "exitosas",
                "errores",
                "tasa_exito",
                "tiempo_promedio_s",
                "tiempo_total_s",
            ]
        ]
        st.caption(
            "Cada fila representa una combinación probada. La tabla se actualiza "
            "al seleccionar una ejecución actual o histórica."
        )
        st.dataframe(comparison, width="stretch", hide_index=True)
        for i, r in enumerate(selected_result, 1):
            icon = "✅" if r["ok"] else "❌"
            with st.expander(
                f"{icon} [{i}] {r['pipeline']} + {r['modelo']} | "
                f"{r['pregunta'][:60]} ({r['tiempo_s']}s)"
            ):
                st.text(r["respuesta"])

        # Export CSV
        st.download_button(
            "📥 Descargar Resultados (CSV)",
            data=df_results.to_csv(index=False).encode("utf-8"),
            file_name="resultados_test_ep2.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.subheader("Historial de ejecuciones de testeo")
    st.caption(
        "Muestra las baterías ejecutadas anteriormente, incluyendo pipeline, "
        "resultado y tiempo total. Se conserva en memoria_larga.json."
    )
    if not test_history:
        st.info("Aún no hay ejecuciones de la batería.")
    else:
        total_runs = len(test_history)
        total_questions = sum(run["total"] for run in test_history)
        total_successes = sum(run["exitosas"] for run in test_history)
        total_errors = sum(run["errores"] for run in test_history)
        total_time = sum(run["tiempo_total_s"] for run in test_history)
        average_time = total_time / total_runs
        success_rate = (total_successes / total_questions * 100) if total_questions else 0
        st.markdown("**Resumen estadístico del testeo**")
        stat_cols = st.columns(4)
        stat_cols[0].metric("Baterías", total_runs)
        stat_cols[1].metric("Preguntas", total_questions)
        stat_cols[2].metric("Tasa de éxito", f"{success_rate:.1f}%")
        stat_cols[3].metric("Tiempo promedio", f"{average_time:.1f}s")
        test_history_df = pd.DataFrame([
            {
                "timestamp": run["timestamp"],
                "pipeline": run["pipeline"],
                "modelo": run.get("modelo", "modelo no registrado"),
                "total": run["total"],
                "exitosas": run["exitosas"],
                "errores": run["errores"],
                "tiempo_total_s": run["tiempo_total_s"],
            }
            for run in test_history
        ])
        st.download_button(
            "📥 Descargar historial de testeo (CSV)",
            data=test_history_df.to_csv(index=False).encode("utf-8"),
            file_name="historial_testeo.csv",
            mime="text/csv",
        )
        for i, run in enumerate(reversed(test_history), 1):
            with st.expander(
                f"{i}. {run['timestamp']} · {run['pipeline']} · "
                f"{run.get('modelo', 'modelo no registrado')} · "
                f"{run['exitosas']}/{run['total']} exitosas · {run['tiempo_total_s']}s"
            ):
                st.write(
                    f"**Preguntas:** {run['total']}  |  "
                    f"**Exitosas:** {run['exitosas']}  |  "
                    f"**Errores:** {run['errores']}  |  "
                    f"**Tiempo:** {run['tiempo_total_s']}s"
                )
                for item in run["resultados"]:
                    icon = "✅" if item["ok"] else "❌"
                    st.markdown(
                        f"{icon} **{item['pregunta']}** — {item['tiempo_s']}s"
                    )
        if st.button("🗑️ Limpiar historial de testeo"):
            st.session_state["test_history"] = []
            if MEMORY_FILE.exists():
                remaining = [
                    item for item in _load_long_term_memory()
                    if "test_run" not in item
                ]
                with MEMORY_FILE.open("w", encoding="utf-8") as file:
                    json.dump(remaining[-20:], file, ensure_ascii=False, indent=2)
            st.rerun()