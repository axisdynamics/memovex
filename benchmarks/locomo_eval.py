"""
memovex — LoCoMo-style Evaluation.

Simulates a long conversational history (60 turns) with facts scattered
at different turn depths, then evaluates recall across three difficulty tiers:

  - Tier 1  Single-hop, recent (turns 40-60)
  - Tier 2  Single-hop, distant (turns 1-20)
  - Tier 3  Cross-turn inference (facts split across 2 turns)

Metrics: Recall@1, Recall@5, MRR, Token-F1.

Reference: Maharana et al. "LoCoMo: Evaluating Long Context Models on
           Long-Term Conversational Memory" (2024).
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from memovex.core.memory_bank import MemoVexOrchestrator
from memovex.core.types import MemoryType


# ---------------------------------------------------------------------------
# Dataset — 60-turn simulated conversation
# ---------------------------------------------------------------------------

CONVERSATION: List[Dict[str, Any]] = [
    # ---- Turns 1-10: personal & project setup ----
    {"turn": 1,  "role": "user",      "text": "Hola, me llamo Matías y soy el lead developer del proyecto."},
    {"turn": 2,  "role": "assistant", "text": "Hola Matías, bienvenido. ¿En qué puedo ayudarte hoy?"},
    {"turn": 3,  "role": "user",      "text": "Estamos construyendo una API REST en Python con FastAPI."},
    {"turn": 4,  "role": "assistant", "text": "Entendido. FastAPI es excelente para APIs de alto rendimiento."},
    {"turn": 5,  "role": "user",      "text": "El stack usa PostgreSQL como base de datos principal."},
    {"turn": 6,  "role": "assistant", "text": "¿Están usando algún ORM como SQLAlchemy?"},
    {"turn": 7,  "role": "user",      "text": "Sí, SQLAlchemy con alembic para las migraciones."},
    {"turn": 8,  "role": "assistant", "text": "Una combinación robusta. ¿Tienen tests?"},
    {"turn": 9,  "role": "user",      "text": "Usamos pytest con cobertura del 87%."},
    {"turn": 10, "role": "assistant", "text": "Excelente cobertura."},
    # ---- Turns 11-20: infra & deployment ----
    {"turn": 11, "role": "user",      "text": "El proyecto corre en Docker y lo desplegamos en AWS ECS."},
    {"turn": 12, "role": "assistant", "text": "¿Usan ECS con Fargate o EC2?"},
    {"turn": 13, "role": "user",      "text": "Fargate, para no gestionar instancias."},
    {"turn": 14, "role": "assistant", "text": "Conveniente. ¿Tienen CI/CD?"},
    {"turn": 15, "role": "user",      "text": "Sí, GitHub Actions. El pipeline hace lint, tests y push a ECR."},
    {"turn": 16, "role": "assistant", "text": "¿Tienen algún SLA de disponibilidad?"},
    {"turn": 17, "role": "user",      "text": "El SLA es 99.5% de uptime mensual."},
    {"turn": 18, "role": "assistant", "text": "¿Usan alguna herramienta de observabilidad?"},
    {"turn": 19, "role": "user",      "text": "DataDog para métricas y alertas, con retención de 15 días."},
    {"turn": 20, "role": "assistant", "text": "Bien. ¿Tienen alertas configuradas?"},
    # ---- Turns 21-30: auth & security ----
    {"turn": 21, "role": "user",      "text": "Las alertas van a Slack al canal #alertas-prod."},
    {"turn": 22, "role": "assistant", "text": "¿Cómo manejan la autenticación?"},
    {"turn": 23, "role": "user",      "text": "JWT con expiración de 24 horas y refresh tokens de 30 días."},
    {"turn": 24, "role": "assistant", "text": "¿Tienen rate limiting?"},
    {"turn": 25, "role": "user",      "text": "Sí, 1000 requests por minuto por API key."},
    {"turn": 26, "role": "assistant", "text": "¿Algún WAF o protección adicional?"},
    {"turn": 27, "role": "user",      "text": "Usamos AWS WAF y tenemos auditoría SOC 2 pendiente para Q3."},
    {"turn": 28, "role": "assistant", "text": "¿Quién es el responsable de seguridad?"},
    {"turn": 29, "role": "user",      "text": "Elena Vargas es la CISO y lidera el programa de seguridad."},
    {"turn": 30, "role": "assistant", "text": "Entendido."},
    # ---- Turns 31-40: performance & incidents ----
    {"turn": 31, "role": "user",      "text": "El endpoint más crítico es GET /api/v2/orders con p99 de 180ms."},
    {"turn": 32, "role": "assistant", "text": "¿Han tenido incidentes recientes?"},
    {"turn": 33, "role": "user",      "text": "En enero tuvimos un outage de 45 minutos por un index faltante en PostgreSQL."},
    {"turn": 34, "role": "assistant", "text": "¿Cómo lo resolvieron?"},
    {"turn": 35, "role": "user",      "text": "Añadimos el index CONCURRENTLY y el tiempo de query bajó de 3s a 12ms."},
    {"turn": 36, "role": "assistant", "text": "¿Tienen connection pooling?"},
    {"turn": 37, "role": "user",      "text": "Sí, PgBouncer con pool_size 50 y max_client_conn 500."},
    {"turn": 38, "role": "assistant", "text": "¿Cuántos requests/seg maneja en producción?"},
    {"turn": 39, "role": "user",      "text": "Pico de 2400 RPS en hora punta, promedio 850 RPS."},
    {"turn": 40, "role": "assistant", "text": "Robusto. ¿Tienen caching?"},
    # ---- Turns 41-50: caching & data ----
    {"turn": 41, "role": "user",      "text": "Redis para cache de sesión y respuestas frecuentes, TTL de 5 minutos."},
    {"turn": 42, "role": "assistant", "text": "¿Qué volumen de datos maneja la DB?"},
    {"turn": 43, "role": "user",      "text": "La tabla de órdenes tiene 120 millones de registros."},
    {"turn": 44, "role": "assistant", "text": "¿Tienen particionamiento?"},
    {"turn": 45, "role": "user",      "text": "Particionamos por fecha de creación, particiones mensuales."},
    {"turn": 46, "role": "assistant", "text": "¿Tienen backup y DR?"},
    {"turn": 47, "role": "user",      "text": "Backups diarios a S3 con retención 90 días y RTO de 4 horas."},
    {"turn": 48, "role": "assistant", "text": "¿Cómo está organizado el equipo?"},
    {"turn": 49, "role": "user",      "text": "El equipo tiene 8 developers: 3 backend, 2 frontend, 2 data engineers y 1 DevOps."},
    {"turn": 50, "role": "assistant", "text": "¿Hacen sprints?"},
    # ---- Turns 51-60: roadmap & tech debt ----
    {"turn": 51, "role": "user",      "text": "Sprints de 2 semanas con planning los lunes y retro los viernes."},
    {"turn": 52, "role": "assistant", "text": "¿Cuáles son las prioridades del Q2?"},
    {"turn": 53, "role": "user",      "text": "Migrar el módulo de pagos de Stripe v2 a v3 y reducir la deuda técnica en el módulo de reportes."},
    {"turn": 54, "role": "assistant", "text": "¿Tienen métricas de deuda técnica?"},
    {"turn": 55, "role": "user",      "text": "SonarCloud marca 340 code smells y 12 bugs críticos pendientes."},
    {"turn": 56, "role": "assistant", "text": "¿Cuándo es el próximo release?"},
    {"turn": 57, "role": "user",      "text": "Release v2.4.0 está planificado para el 15 de mayo con feature flags habilitados."},
    {"turn": 58, "role": "assistant", "text": "¿Tienen documentación de la API?"},
    {"turn": 59, "role": "user",      "text": "Swagger en /docs y Postman collection compartida internamente."},
    {"turn": 60, "role": "assistant", "text": "Perfecto, tienes todo muy bien organizado Matías."},
]

# ---------------------------------------------------------------------------
# Evaluation queries — three tiers
# ---------------------------------------------------------------------------

QUERIES: List[Dict[str, Any]] = [
    # Tier 1: Recent facts (turns 40-60)
    {"id": "L01", "tier": 1, "question": "¿Qué base de datos se usa para cache y cuál es el TTL?",
     "answer": "Redis TTL 5 minutos", "keywords": ["redis", "ttl", "5"],
     "source_turns": [41]},
    {"id": "L02", "tier": 1, "question": "¿Cuántos code smells reporta SonarCloud?",
     "answer": "340", "keywords": ["340", "sonarcloud", "code smells"],
     "source_turns": [55]},
    {"id": "L03", "tier": 1, "question": "¿Cuándo está planificado el release v2.4.0?",
     "answer": "15 de mayo", "keywords": ["mayo", "2.4.0", "release"],
     "source_turns": [57]},
    {"id": "L04", "tier": 1, "question": "¿Cuántos registros tiene la tabla de órdenes?",
     "answer": "120 millones", "keywords": ["120", "millones", "órdenes"],
     "source_turns": [43]},
    {"id": "L05", "tier": 1, "question": "¿Cuántos developers tiene el equipo?",
     "answer": "8 developers", "keywords": ["8", "developers", "equipo"],
     "source_turns": [49]},

    # Tier 2: Distant facts (turns 1-20)
    {"id": "L06", "tier": 2, "question": "¿Cómo se llama el lead developer del proyecto?",
     "answer": "Matías", "keywords": ["matías", "lead", "developer"],
     "source_turns": [1]},
    {"id": "L07", "tier": 2, "question": "¿Qué framework web usa el proyecto?",
     "answer": "FastAPI", "keywords": ["fastapi", "python", "api"],
     "source_turns": [3]},
    {"id": "L08", "tier": 2, "question": "¿Cuál es la cobertura de tests?",
     "answer": "87%", "keywords": ["87", "cobertura", "pytest"],
     "source_turns": [9]},
    {"id": "L09", "tier": 2, "question": "¿En qué plataforma cloud está desplegado el proyecto?",
     "answer": "AWS ECS Fargate", "keywords": ["aws", "ecs", "fargate"],
     "source_turns": [11, 13]},
    {"id": "L10", "tier": 2, "question": "¿Cuál es el SLA de disponibilidad?",
     "answer": "99.5%", "keywords": ["99.5", "sla", "uptime"],
     "source_turns": [17]},

    # Tier 3: Cross-turn inference (need to combine facts)
    {"id": "L11", "tier": 3, "question": "¿Cuánto tiempo expiran los JWT tokens de acceso?",
     "answer": "24 horas", "keywords": ["24", "horas", "jwt"],
     "source_turns": [23]},
    {"id": "L12", "tier": 3, "question": "¿Qué causó el outage de enero y cómo se resolvió?",
     "answer": "index faltante resolvió CONCURRENTLY 12ms", "keywords": ["index", "concurrently", "12ms"],
     "source_turns": [33, 35]},
    {"id": "L13", "tier": 3, "question": "¿Quién lidera el programa de seguridad y qué auditoría tienen pendiente?",
     "answer": "Elena Vargas SOC 2 Q3", "keywords": ["elena", "vargas", "soc"],
     "source_turns": [27, 29]},
    {"id": "L14", "tier": 3, "question": "¿Cuántos RPS maneja el sistema en hora punta?",
     "answer": "2400 RPS", "keywords": ["2400", "rps", "punta"],
     "source_turns": [39]},
    {"id": "L15", "tier": 3, "question": "¿Cuál es el pool_size de PgBouncer y el max_client_conn?",
     "answer": "pool_size 50 max_client_conn 500", "keywords": ["50", "500", "pgbouncer"],
     "source_turns": [37]},
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = set(_tokenize(prediction))
    gt_tokens = set(_tokenize(ground_truth))
    if not gt_tokens:
        return 1.0
    if not pred_tokens:
        return 0.0
    overlap = pred_tokens & gt_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_hit(retrieved_text: str, keywords: List[str]) -> bool:
    """True if any keyword appears in the retrieved text."""
    low = retrieved_text.lower()
    return any(kw.lower() in low for kw in keywords)


def mrr(hits_list: List[List[bool]]) -> float:
    """Mean Reciprocal Rank given per-query hit lists."""
    rrs = []
    for hits in hits_list:
        rr = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(rrs) / max(len(rrs), 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def seed_conversation(bank: MemoVexOrchestrator) -> None:
    for entry in CONVERSATION:
        role_tag = "user_turn" if entry["role"] == "user" else "assistant_turn"
        bank.store(
            text=f"[Turn {entry['turn']}][{entry['role']}] {entry['text']}",
            memory_type=MemoryType.EPISODIC,
            tags={role_tag, f"turn_{entry['turn']}"},
            confidence=0.8 if entry["role"] == "user" else 0.7,
            salience=0.65 if entry["role"] == "user" else 0.5,
        )


def evaluate(bank: MemoVexOrchestrator,
             queries: List[Dict],
             k_values: Tuple[int, ...] = (1, 3, 5)) -> Dict[str, Any]:
    recall_at_k = {k: 0 for k in k_values}
    f1_scores: List[float] = []
    hits_per_query: List[List[bool]] = []

    results_detail = []

    for q in queries:
        retrieved = bank.retrieve(q["question"], top_k=max(k_values))
        texts = [r.memory.text for r in retrieved]

        # Per-rank hits (keyword match)
        hit_list = [keyword_hit(t, q["keywords"]) for t in texts]
        hits_per_query.append(hit_list)

        # Recall@k
        for k in k_values:
            if any(hit_list[:k]):
                recall_at_k[k] += 1

        # Token F1 — best match across top-5
        best_f1 = max((token_f1(t, q["answer"]) for t in texts[:5]), default=0.0)
        f1_scores.append(best_f1)

        results_detail.append({
            "id": q["id"],
            "tier": q["tier"],
            "hit@1": hit_list[0] if hit_list else False,
            "hit@5": any(hit_list[:5]),
            "f1": round(best_f1, 3),
            "top_text": texts[0][:80] if texts else "",
        })

    n = len(queries)
    return {
        **{f"recall@{k}": round(recall_at_k[k] / n, 3) for k in k_values},
        "mrr": round(mrr(hits_per_query), 3),
        "token_f1": round(sum(f1_scores) / max(len(f1_scores), 1), 3),
        "n_queries": n,
        "detail": results_detail,
    }


def run_locomo(embeddings_enabled: bool = False, verbose: bool = True) -> Dict[str, Any]:
    bank = MemoVexOrchestrator(agent_id="locomo_eval",
                                  embeddings_enabled=embeddings_enabled)
    bank.initialize()

    t0 = time.time()
    seed_conversation(bank)
    seed_ms = (time.time() - t0) * 1000

    # Full evaluation
    all_metrics = evaluate(bank, QUERIES)

    # Per-tier breakdown
    tier_metrics = {}
    for tier in (1, 2, 3):
        tier_qs = [q for q in QUERIES if q["tier"] == tier]
        tier_metrics[f"tier{tier}"] = evaluate(bank, tier_qs)

    # Latency
    latencies = []
    for q in QUERIES:
        ts = time.time()
        bank.retrieve(q["question"], top_k=5)
        latencies.append((time.time() - ts) * 1000)
    avg_lat = round(sum(latencies) / len(latencies), 1)
    p99_lat = round(sorted(latencies)[int(len(latencies) * 0.99)], 1)

    bank.shutdown()

    result = {
        "benchmark": "LoCoMo-style",
        "turns": len(CONVERSATION),
        "queries": len(QUERIES),
        "seed_ms": round(seed_ms, 1),
        "avg_latency_ms": avg_lat,
        "p99_latency_ms": p99_lat,
        "embeddings": embeddings_enabled,
        **all_metrics,
        "tier_breakdown": tier_metrics,
    }

    if verbose:
        _print_locomo(result)

    return result


def _print_locomo(r: Dict[str, Any]) -> None:
    print("\n" + "=" * 58)
    print(f"  LoCoMo-style Evaluation — MemoVex")
    print(f"  Turns: {r['turns']}  |  Queries: {r['queries']}")
    print("=" * 58)
    print(f"  Recall@1    {r['recall@1']:.3f}")
    print(f"  Recall@3    {r['recall@3']:.3f}")
    print(f"  Recall@5    {r['recall@5']:.3f}")
    print(f"  MRR         {r['mrr']:.3f}")
    print(f"  Token-F1    {r['token_f1']:.3f}")
    print(f"  Latency     {r['avg_latency_ms']:.1f} ms avg  /  {r['p99_latency_ms']:.1f} ms p99")
    print()
    for tier_key, tm in r["tier_breakdown"].items():
        label = {"tier1": "Tier 1 (recent)", "tier2": "Tier 2 (distant)", "tier3": "Tier 3 (cross-turn)"}[tier_key]
        print(f"  {label:25s}  R@1={tm['recall@1']:.3f}  R@5={tm['recall@5']:.3f}  F1={tm['token_f1']:.3f}")
    print()
    for d in r.get("detail", []):
        status = "✓" if d["hit@1"] else ("~" if d["hit@5"] else "✗")
        print(f"  [{status}] {d['id']} (T{d['tier']})  F1={d['f1']:.2f}  {d['top_text'][:55]}")
    print("=" * 58)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", action="store_true")
    args = p.parse_args()
    run_locomo(embeddings_enabled=args.embeddings)
