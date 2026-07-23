# Plano: migrar guardrail_eval para RunPod (GPU) + auditoria automatizada (A.22)

> Status: **código implementado** (Fase 3 — ver `ARCHITECTURE.md` e os
> módulos `ground_truth.py` / `audit_agent.py` / `run_audit_pipeline.py`).
> Falta **executar**: primeiro o dry-run local em CPU, depois o smoke maior
> no RunPod. Este arquivo é o guia de execução; o desenho detalhado está no
> `ARCHITECTURE.md` (seção Phase 3). Nada em `jlens/` foi alterado.
>
> **Antes de rodar:** criar `.env` na raiz do repo com
> `GEMINI_API_KEY=<sua chave do Google AI Studio>` (o `.env` já está no
> `.gitignore`) e instalar as deps novas:
> `pip install -r guardrail_eval/requirements.txt` (adiciona `google-genai`
> e `python-dotenv`).
>
> **Dry-run local (CPU, sem custo):**
> `python run_audit_pipeline.py --device cpu --n-malign 2 --n-benign 2`
> e inspecionar `results_audit/audit_scores.jsonl`.

## Context

O `guardrail_eval/` hoje roda 100% CPU (~7.7GB RAM), o que forçou um design
de carregamento sequencial de modelos (nunca guardrail+target juntos) e
mantém tudo em escala "smoke" (poucos prompts, ~2min/prompt). O próximo
passo do projeto é migrar para uma GPU no RunPod e começar a implementar a
arquitetura do Apêndice A.22 do paper — um agente investigador com acesso à
ferramenta de readout J-lens, avaliado por um LLM-judge contra um ground
truth — aplicada ao guardrail Qwen3-1.7B já existente, no espírito do
`markdowns-de-referencia/automacao_auditoria_jlens.md` (já avaliado
anteriormente: viável em escala pequena, com a Estratégia B causal
bloqueada por primitivas de ablation/swap que não existem em `jlens` nem no
sub-projeto).

Decisões de escopo já fechadas para este primeiro smoke test:

- **Só o loop de auditoria** (guardrail + J-lens + investigador + judge) —
  **sem** o target model (`gemma-3-1b-it`) por enquanto. Fiel ao A.22, que
  não envolve modelo alvo.
- **Ground truth: Estratégia A (comportamental)** — cruza o rótulo do
  dataset com o veredito do guardrail (ex: seed malign + guardrail disse
  benign = falso negativo silencioso). Não depende de ablation/swap.
- **Investigador + judge: Gemini** (Google AI Studio, free tier), mesmo
  provedor para os dois papéis por enquanto.
- **Carregamento de modelo: simplificado** — como só o guardrail é usado
  neste loop (sem target), a dança de duas fases do
  `run_attack_pipeline.py` nem se aplica aqui; o novo driver carrega o
  guardrail uma única vez. Quando o target voltar a fazer parte do escopo,
  a GPU tem folga (~11GB fp32 pros dois juntos vs. 16-24GB de VRAM) para
  carregá-los junto — anotado como nota futura, não implementado agora.

`jlens` em si já é GPU/dtype-agnóstico por design: `from_hf()` documenta
explicitamente que o modelo já deve chegar "on the target device and
dtype" — o resto do código (`HFLensModel.encode`/`.unembed`,
`JacobianLens.transport`) já move tensores via `.to(residual.device)`
automaticamente. **Não é necessário tocar em `jlens/`** — só em
`guardrail_eval/` para passar `device=` na hora de instanciar o modelo HF.

## O que muda em `guardrail_eval/`

### 1. `jlens_readout.py` — suporte a device na `GuardrailLens`

Adicionar parâmetro `device: str = "cpu"` ao `__init__`; após
`AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)`, mover com
`.to(device)` antes de `.eval()`/`jlens.from_hf(...)`. `chat_prompt`,
`classify`, `readout` não precisam mudar — já são device-agnósticos
(`model.encode()` resolve o device via `HFLensModel.input_device`, e
`JacobianLens.apply()` já devolve tudo em `.cpu()`).

### 2. Novo: `ground_truth.py` — Estratégia A

Duas claims (categorias do §5 do `automacao_auditoria_jlens.md`) escolhidas
por serem diretamente computáveis por cruzamento rótulo×veredito, sem
heurística de keyword nem dado causal:

- **Falso negativo silencioso**: `label_true == "malign" and guardrail_verdict == "benign"`.
- **Especificidade / falso positivo**: `label_true == "benign" and guardrail_verdict == "malign"`.

As outras 4 categorias do markdown (momento de emergência, deliberação
interna, robustez a disfarce, reconhecimento silencioso genérico) exigiriam
ou heurística de keyword sobre o readout ou dados causais — fora de escopo
deste smoke test, documentado como extensão futura.

Reaproveita a lógica de rótulo/verdito que já existe em
`run_attack_pipeline.py::run_guardrail_phase` (campos `label_true` /
`guardrail_label_pred` do JSONL) — não precisa reimplementar a
classificação, só consumir o resultado já produzido pela `GuardrailLens`.

### 3. Novo: `audit_agent.py` — investigador + judge via Gemini

- Wrap de `GuardrailLens.readout()` como *tool* no formato de function
  calling do Gemini (`google-generativeai` SDK): nome `get_jlens_readout`,
  parâmetros `layers: list[int] | None`, `top_k: int` — posição fixa na
  posição de decisão (`-1`), já que o sweep de posições (momento de
  emergência) está fora de escopo.
- Loop manual de tool-calling: manda transcript + claim pro Gemini com
  `tools=[...]`; se ele responder com uma function call, executa
  `gl.readout(...)` localmente e devolve o resultado como
  `function_response`; repete até ele responder com veredito final em
  texto (evidência citada + claim resolvida sim/não).
- Judge: segunda chamada Gemini (prompt separado, sem tool access),
  recebe `(claim, resposta do investigador, ground truth)`, devolve score
  0-10 + justificativa curta.
- Carrega a API key via `python-dotenv` a partir do `.env` da raiz do
  repo (`GEMINI_API_KEY=...`).

### 4. Novo: `run_audit_pipeline.py` — driver

Segue o padrão de CLI de `run_attack_pipeline.py` (`--n-malign`,
`--n-benign`, `--device`, `--dtype`, `--top-k`, `--layers`, `--verbose`),
lendo de `data/attack_baseline.csv` (seeds sem wrapping — wrapping fica
para quando "robustez a disfarce" entrar em escopo). Por prompt: roda
`gl.classify()` + `gl.readout()` (já existente), computa ground truth
(`ground_truth.py`) para as 2 claims, roda investigador+judge
(`audit_agent.py`) para cada claim, grava:

- `results/audit_readouts.jsonl` — um registro por prompt (reaproveita o
  formato usado em `run_attack_pipeline.py`, com `jlens` embutido).
- `results/audit_scores.jsonl` — um registro por (prompt, claim): ground
  truth, resposta do investigador, evidência citada, score do judge.
- `results/audit_summary.csv` — agregado por categoria de claim (média de
  score, contagem), no espírito da seção 6 do markdown.

Default de smoke: `--n-malign 5 --n-benign 5` (10 prompts × 2 claims = 20
ciclos investigador+judge). Em GPU isso deve rodar em bem menos que os
20-30 min de orçamento (guardrail: <2s/prompt; investigador+judge: ~10-20s
por claim via API) — dá margem para subir N se sobrar tempo.

### 5. `requirements.txt` / `.gitignore`

- Adicionar `google-generativeai`, `python-dotenv` a
  `guardrail_eval/requirements.txt`.
- **Adicionar `.env` ao `.gitignore` da raiz** — hoje não está protegido,
  e vai passar a guardar `GEMINI_API_KEY`. Corrigir antes de qualquer
  chave real entrar no arquivo.

### 6. `target_model.py` (touch mínimo, preparo futuro)

Adicionar o mesmo parâmetro `device` que `GuardrailLens` recebe, e mover
`encoded` (dict do `apply_chat_template`) pro device antes de
`self.hf.generate(**encoded, ...)` — hoje ele fica implicitamente em CPU
mesmo que o modelo esteja em GPU. Não integrado ao `run_audit_pipeline.py`
agora (target fora de escopo), mas deixa `target_model.py` pronto pra
quando o target voltar a entrar no pipeline.

## Setup do RunPod (execução manual)

Execução planejada via **VSCode Remote-SSH** direto no pod (editor +
terminal integrado rodando no host remoto).

1. Pod GPU: **RTX 4090 ou A5000 (24GB VRAM), Community Cloud**, template
   oficial "PyTorch" (CUDA já vem instalado — evita mismatch de driver),
   ~30-40GB de container disk. Confirmar na criação do pod que **SSH
   Terminal Access** está habilitado (alguns templates só dão terminal
   web, sem `ssh` de linha de comando — não serve pro Remote-SSH) e que
   sua chave pública SSH está cadastrada na conta RunPod.
2. Conectar o VSCode via Remote-SSH ao pod (host/porta/chave que o RunPod
   fornece no dashboard). A partir daí, abrir o terminal integrado
   (rodando no pod) e `git clone` o repo de lá — preservando a estrutura
   atual (`guardrail_eval/` como sub-projeto, nunca mexendo em `jlens/` /
   `tests/` / `data/` / `harmbench.csv` da raiz).
3. `pip install -e ..` (jlens) + `pip install -r guardrail_eval/requirements.txt`
   dentro do venv do pod (o template já traz torch com CUDA; não reinstalar
   torch).
4. Levar o `.env` com `GEMINI_API_KEY` pro pod manualmente (arrastar pelo
   explorer remoto do VSCode ou colar o conteúdo direto num arquivo criado
   no pod) — nunca via git, já que não estava no `.gitignore` até agora.
5. Rodar `python run_audit_pipeline.py --device cuda --dtype bfloat16
   --n-malign 5 --n-benign 5` (pelo terminal integrado do VSCode) e
   inspecionar `results/audit_scores.jsonl`.
6. Derrubar o pod depois de validar (billing por minuto) — encerrar a
   sessão Remote-SSH e terminar o pod pelo dashboard do RunPod.

## Verificação

- Checar `torch.cuda.is_available()` no pod antes de rodar qualquer coisa.
- Rodar `run_audit_pipeline.py` com N bem pequeno primeiro (ex.
  `--n-malign 2 --n-benign 2`) e inspecionar manualmente 1-2 linhas de
  `audit_scores.jsonl` — confirmar que o investigador está de fato citando
  evidência do readout (não alucinando) e que o judge está pontuando de
  forma coerente com o ground truth.
- Cronometrar essa rodada pequena antes de subir para o `--n-malign 5
  --n-benign 5` default, para confirmar que cabe no orçamento de 20-30 min.
- Conferir que `guardrail_eval/data/`, `jlens/`, `tests/`, `data/` e
  `harmbench.csv` da raiz permanecem intocados (mesma restrição de sempre
  do sub-projeto).
