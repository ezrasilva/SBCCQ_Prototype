---
description: "Estruturar/atualizar os 3 experimentos (Threshold, On-Demand, Hybrid) com 30 repetições, seeds reproduzíveis, CSV e gráficos"
name: "Estruturar Experimentos QKD"
argument-hint: "Ajuste: (exp=1|2|3) e/ou parâmetros (reps, seeds, perfis)"
agent: "agent"
model: "GPT-5 (copilot)"
---

Você está em um repositório que simula gerenciamento de chaves QKD com políticas `threshold`, `on_demand` e `hybrid`.

Tarefa: estruturar (ou revisar) os experimentos do artigo em 3 blocos:

1) Experimento 1 — Comparação básica
- Mesma configuração base para todas as políticas
- Parâmetros fixos
- 30 repetições independentes por política com seeds fixas e reproduzíveis
- Saídas: CSV por repetição + CSV de resumo (média/desvio) + gráficos comparativos por política
- Métricas principais: `service_rate`, `denial_rate`, `buffer_util_mean`, `replenishment_events`

2) Experimento 2 — Carga moderada, sobrecarga e burst
- Mesmo cenário estrutural, variando apenas padrão de chegada
- Incluir 3 perfis no mesmo bloco: moderada, sobrecarga e burst (distribuição temporal diferente)
- 30 repetições por política e perfil
- Saídas: CSVs + gráficos comparativos por política e por perfil

3) Experimento 3 — Sensibilidade do limiar
- Variar limiar em uma faixa (ex.: baixo/médio/alto)
- Aplicar apenas às políticas que dependem do limiar (`threshold` e `hybrid`)
- Carga fixa representativa
- 30 repetições por política e valor de limiar
- Saídas: CSVs + gráficos (curvas com barras de erro)

Regras obrigatórias:
- Seeds determinísticas (não usar `hash()` de Python para seed)
- Exportar tabelas em CSV e também calcular média e desvio padrão
- Gerar gráficos comparativos por política

Ao executar a tarefa:
- Reaproveite o módulo de experimentos em `quantumnet/experiments/qkd_policy_experiments.py` quando existir
- Se precisar adicionar uma métrica, preferir estender `get_qkd_link_state()` e o agregador de experimentos, sem quebrar compatibilidade
- Responda com: (1) arquivos alterados, (2) o que mudou, (3) como executar e onde achar os CSVs/plots.
