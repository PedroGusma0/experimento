# Plano: migrar guardrail_eval para RunPod (GPU) + auditoria automatizada (A.22)

> Status: **código implementado** (Fase 3 — ver `../markdowns-de-referencia/ARCHITECTURE.md` e os
> módulos `ground_truth.py` / `audit_agent.py` / `run_audit_pipeline.py`).
> Dry-run local em CPU já validado. Falta **executar a run principal** (60
> prompts: 15 malign + 15 benign por ataque, baseline + wrapping) no RunPod
> — ver seção "Run principal" abaixo. O desenho detalhado está no
> `../markdowns-de-referencia/ARCHITECTURE.md` (seção Phase 3). Nada em `jlens/` foi alterado.
>
> **Antes de rodar:** criar `.env` na raiz do repo com
> `DEEPSSEK_API_KEY=<chave nvapi-...>` (investigador, via NVIDIA) e
> `GPT_API_KEY=<chave>` (judge, via Groq) — o `.env` já está no
> `.gitignore`. Instalar as deps com `bash guardrail_eval/setup_pod.sh` (ver
> seção "Setup do RunPod" — resolve automaticamente incompatibilidade de
> torch/torchvision/torchaudio antes de instalar `jlens` + o resto de
> `requirements.txt`).
>
> **Dry-run local (CPU, sem custo):**
> `python run_audit_pipeline.py --device cpu --attack baseline --n-malign 2 --n-benign 2`
> e inspecionar `results_audit/audit_scores_baseline.jsonl`.

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
- **Investigador: Deepseek** (free tier)
  - **judge: GPT** (free tier)
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

### 3. Novo: `audit_agent.py` —

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
`--n-benign`, `--device`, `--dtype`, `--top-k`, `--layers`, `--verbose`,
mais `--attack {baseline,baseline-wrapping,both}` — cobre os dois ataques,
não só baseline). Por prompt: roda `gl.classify()` + `gl.readout()` (já
existente), computa ground truth (`ground_truth.py`) para as 2 claims, roda
investigador+judge (`audit_agent.py`) para cada claim, grava (por ataque,
em `results_audit/`, separado do `results/` da Fase 2):

- `results_audit/audit_readouts_<attack>.jsonl` — um registro por prompt
  (reaproveita o formato usado em `run_attack_pipeline.py`, com `jlens`
  embutido).
- `results_audit/audit_scores_<attack>.jsonl` — um registro por (prompt,
  claim): ground truth, resposta do investigador, evidência citada, score
  do judge.
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

1. Pod GPU real desta run: **RTX A4000 (16GB VRAM)**, 18 vCPU (AMD EPYC
   7702), 71GB RAM, 40GB container disk, imagem
   `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (CUDA 12.4.1,
   torch 2.4.0 de fábrica — ver "Dependências do pod" abaixo pro porquê
   disso importa). 16GB de VRAM é folga suficiente pro guardrail (~8GB
   gemma-3-4b-it em bf16). Confirmar na criação do pod que **SSH Terminal
   Access** está habilitado (alguns templates só dão terminal web, sem
   `ssh` de linha de comando — não serve pro Remote-SSH) e que sua chave
   pública SSH está cadastrada na conta RunPod.
2. Conectar o VSCode via Remote-SSH ao pod (host/porta/chave que o RunPod
   fornece no dashboard). A partir daí, abrir o terminal integrado
   (rodando no pod) e `git clone` o repo de lá — preservando a estrutura
   atual (`guardrail_eval/` como sub-projeto, nunca mexendo em `jlens/` /
   `tests/` / `data/` / `harmbench.csv` da raiz).
3. Rodar `bash guardrail_eval/setup_pod.sh` — resolve automaticamente
   qualquer incompatibilidade de torch/torchvision/torchaudio (ver seção
   "Dependências do pod" abaixo) e instala `jlens` (`pip install -e ..`) +
   `guardrail_eval/requirements.txt` em sequência. Não rodar os `pip
   install` manualmente — o script já cobre isso de forma idempotente.
4. Levar o `.env` com `DEEPSSEEK_API_KEY` (investigador) e `GPT_API_KEY`
   (judge) pro pod manualmente (arrastar pelo explorer remoto do VSCode ou
   colar o conteúdo direto num arquivo criado no pod) — nunca via git, já
   que não estava no `.gitignore` até pouco tempo atrás.
5. Rodar a **run principal** (ver seção dedicada abaixo para o rollout em
   2 estágios recomendado antes de ir direto pros 60 prompts), **trocando
   o guardrail para gemma-3-4b-it** (via `--guardrail-model`; o preset já
   resolve a lente e os quirks de carregamento):
   ```
   python run_audit_pipeline.py --guardrail-model google/gemma-3-4b-it \
       --device cuda --dtype bfloat16 --attack both \
       --n-malign 15 --n-benign 15 --resume --verbose
   ```
   e inspecionar `results_audit/audit_scores_baseline.jsonl`,
   `results_audit/audit_scores_baseline_wrapping.jsonl` e
   `results_audit/audit_summary_combined.csv`.
6. Derrubar o pod depois de validar (billing por minuto) — encerrar a
   sessão Remote-SSH e terminar o pod pelo dashboard do RunPod.

### Dependências do pod (a causa de uma sessão de debug anterior)

O template `runpod/pytorch:2.4.0-...-cuda12.4.1-...` traz torch 2.4.0, mas
`transformers>=5.5` (exigido pelo `jlens`) — especificamente a 5.14.1 —
importa `torch.distributed.tensor.DTensor`, que não existe no torch 2.4.0
(`ImportError`). Numa sessão anterior, corrigir isso rodando `pip install
--upgrade torch` sozinho quebrou `torchvision`/`torchaudio` em cadeia (ABI
incompatível: `operator torchvision::nms does not exist`, depois `undefined
symbol` no `libtorchaudio.so`) — cada fix isolado de uma lib quebrava a
próxima. **`guardrail_eval/setup_pod.sh` automatiza a correção certa**:
testa se `DTensor`/`torchvision`/`torchaudio` importam; se não, reinstala os
três **juntos, de um único índice CUDA-casado**
(`pip install --upgrade torch torchvision torchaudio --index-url
https://download.pytorch.org/whl/cu124`, casando com o CUDA 12.4.1 do pod)
em vez de atualizar um de cada vez. Se ainda assim falhar, o script sai com
erro e diagnóstico em vez de mascarar o problema.

### Troca do guardrail para gemma-3-4b-it (a fazer no pod)

O código já suporta a troca via `PRESETS` em `jlens_readout.py` (basta
`--guardrail-model google/gemma-3-4b-it`). Lente confirmada existente:
`gemma-3-4b-it/jlens/Salesforce-wikitext/gemma-3-4b-it_jacobian_lens.pt` no
mesmo repo `neuronpedia/jacobian-lens`. Diferenças em relação ao Qwen3 já
tratadas no preset: carregamento via `AutoModelForImageTextToText`
(gemma-3-4b é multimodal, fora do auto-map de CausalLM), `force_bos=True`,
sem `enable_thinking`. **Só é rodável na GPU** (~8GB em bf16, não cabe nos
7.7GB de RAM da máquina local — por isso a troca é exclusiva do pod).

**Verificar no PRIMEIRO run no pod (não testável localmente):**
- Se `AutoModelForImageTextToText` carrega e o `jlens.from_hf` acha o text
  decoder (layout `language_model`). Se falhar, passar `layout=` explícito.
- Se o chat template do gemma-3 aceita `role: "system"`. O template do
  gemma-2 **não** aceitava (levantava erro). Se o gemma-3 também não aceitar,
  dobrar o `SYSTEM_PROMPT` no início do turno `user` em `chat_prompt`.
- Que a posição de decisão (`-1`) e a workspace band fazem sentido para o
  gemma-3-4b (28 camadas de texto? conferir `source_layers` da lente e
  ajustar `--layers` se necessário — o default L14-26 foi calibrado no Qwen3).

### Investigador multi-posição + tool-calling interativo (a fazer no pod)

O investigador agora sonda a J-lens interativamente (`get_jlens_readout`,
formato function-calling OpenAI-compatible) em vez de receber um único
readout pronto — ver `ARCHITECTURE.md` (seção Phase 3) e
`audit_agent.py::investigate`. Também não testável localmente (o loop depende
do endpoint DeepSeek/NVIDIA aceitar `tools`).

**Verificar no PRIMEIRO run no pod:**
- Se o endpoint `deepseek-ai/deepseek-v4-pro` (via NVIDIA) aceita
  `tools`/function-calling de fato. Se não aceitar, `investigate()` já cai
  automaticamente para `_investigate_fixed` (leitura fixa em `-1`) — conferir
  em `audit_scores_<attack>.jsonl` se `fallback_used=true` nesse caso (não é
  erro, é o comportamento esperado de degradação).
- Quantas chamadas de tool o investigador usa até convergir (campo
  `tool_calls` em `audit_scores_<attack>.jsonl`) — calibrar `--max-tool-calls`
  (default 5, um chute inicial) para cima ou para baixo com base nisso.
- Se a janela de `--token-span-last-n 60` (últimos N tokens mostrados no mapa
  de posições) é suficiente para cobrir o `seed` inteiro nos prompts reais, ou
  se seeds mais longos precisam de uma janela maior.
- Se as posições que o investigador escolhe fazem sentido (ler
  `investigator_evidence` em `audit_scores_<attack>.jsonl` cruzado com o mapa
  de tokens) — é um mecanismo de puro julgamento do agente (decisão tomada
  nesta conversa: sem pré-anotação, sem fallback determinístico de posição),
  então vale checar qualitativamente se ele está escolhendo pontos
  razoáveis, não só se roda sem erro.

### Run principal (60 prompts: 15 malign + 15 benign por ataque)

Primeira run em escala "principal", reduzida de propósito por preocupação
de free tier — não é o corpus completo (200 malign + 30 benign por ataque =
460 prompts), que fica anotado como possível próximo passo se o free tier
aguentar. Cobre os dois ataques (`baseline` + `baseline-wrapping`), dando
pela primeira vez volume suficiente pra capturar casos de falso negativo
(raros nos smokes de 1-2 prompts feitos até agora).

**Custo de API esperado**: cada linha do corpus dispara no máximo 1 claim
(o veredito do guardrail só pode ser malign OU benign, nunca os dois — visto
em `ground_truth.py`), e cada claim custa entre ~3 e ~7 chamadas de API
(investigador: 1 rodada inicial + até `--max-tool-calls` tool-calls + 1
finalize; judge: 1 chamada). Pra 60 prompts, isso é **~180-420 chamadas de
API no total**, entre DeepSeek e Groq — gerenciável, mas ainda vale rodar em
estágios em vez de disparar direto.

**Rollout recomendado (2 estágios):**
1. Confirmar que nada regrediu com o smoke de sempre:
   `--n-malign 5 --n-benign 5 --attack baseline` (sem `--resume`, sem
   custo de tempo relevante).
2. A run principal em si:
   `--attack both --n-malign 15 --n-benign 15 --resume --verbose`
   (comando completo na seção "Setup do RunPod" acima). `--resume` fica
   disponível caso o pod caia no meio — só rodar o mesmo comando de novo
   retoma sem duplicar linhas nem reprocessar o que já tem `pool_index` em
   `audit_readouts_<attack>.jsonl`. Se aparecerem 429 recorrentes, ajustar
   `--api-pacing-seconds` (dorme entre cada ciclo investigate+judge) e/ou
   baixar `--max-tool-calls` antes de tentar de novo.

## Verificação

- Rodar `bash guardrail_eval/setup_pod.sh` primeiro e confirmar que termina
  sem erro (self-test de DTensor/torchvision/torchaudio passa) antes de
  instalar mais nada.
- Checar `torch.cuda.is_available()` no pod antes de rodar qualquer coisa
  (o próprio `setup_pod.sh` já faz isso no sanity check final).
- Rodar `run_audit_pipeline.py` com N bem pequeno primeiro (ex.
  `--attack baseline --n-malign 2 --n-benign 2`) e inspecionar manualmente
  1-2 linhas de `audit_scores_baseline.jsonl` — confirmar que o investigador
  está de fato citando evidência do readout (não alucinando) e que o judge
  está pontuando de forma coerente com o ground truth.
- Cronometrar essa rodada pequena antes de subir para a run principal
  (`--attack both --n-malign 15 --n-benign 15`, ver seção dedicada acima),
  pra ter uma noção de tempo/custo real antes de comprometer com os 60
  prompts.
- Testar `--resume` deliberadamente: interromper a run principal no meio
  (Ctrl+C), rodar de novo com `--resume` e confirmar que retoma sem duplicar
  linhas nem reprocessar o que já tinha `pool_index` em
  `audit_readouts_<attack>.jsonl`.
- Se aparecerem 429 recorrentes, ajustar `--api-pacing-seconds` e/ou
  `--max-tool-calls` antes de tentar de novo — não simplesmente re-rodar
  esperando sorte.
- Conferir que `guardrail_eval/data/`, `jlens/`, `tests/`, `data/` e
  `harmbench.csv` da raiz permanecem intocados (mesma restrição de sempre
  do sub-projeto).
