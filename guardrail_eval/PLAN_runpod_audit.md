# Plano: migrar guardrail_eval para RunPod (GPU) + auditoria automatizada (A.22)

> Status: **código implementado** (Fase 3 — ver `../markdowns-de-referencia/ARCHITECTURE.md` e os
> módulos `ground_truth.py` / `audit_agent.py` / `run_audit_pipeline.py`).
> Dry-run local em CPU já validado. Falta **executar a run principal** no
> RunPod — ver seção "Run principal" abaixo. O desenho detalhado está no
> `../markdowns-de-referencia/ARCHITECTURE.md` (seção Phase 3). Nada em `jlens/` foi alterado.
>
> **Decisão nova (ainda não implementada, ver seções "Modelos abertos locais
> pro investigador/judge" e "Metadata de proveniência" abaixo): a run no
> RunPod passa a rodar investigador e judge como modelos open-weight locais
> (via `transformers` puro, sem vLLM), não mais via API DeepSeek/Groq.** O
> dry-run local em CPU continua exatamente como está — via API — sem
> nenhuma mudança. Motivação: os testes via API estavam lentos demais pra
> iterar, e vLLM (a alternativa óbvia pra acelerar) traria riscos próprios
> (não-determinismo por batching contínuo, incerteza sobre suporte real a
> tool-calling/JSON mode por modelo+versão). Guardrail + investigador +
> judge continuam residentes juntos o tempo todo, como hoje — sem
> separação de fases.
>
> **Antes de rodar (path API / dry-run CPU):** criar `.env` na raiz do repo com
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
- **Investigador e judge, via API (DeepSeek free tier / Groq free tier)**
  — esse é o path usado pelo **dry-run local em CPU**, que continua
  inalterado. **A run no RunPod troca os dois por modelos open-weight
  locais** — ver "Modelos abertos locais pro investigador/judge" abaixo.
  Os dois paths ficam selecionáveis independentemente via
  `--investigator-backend {api,local}` / `--judge-backend {api,local}`
  (cada um default `api`), não uma troca global — dá pra testar
  combinações mistas (ex. investigador local + judge via API) antes de
  comprometer com os dois locais.
- **Carregamento de modelo: continua único** — guardrail (+ agora
  investigador e judge, quando `--*-backend local`) carregam juntos, uma
  única vez, sem dança de duas fases. Isso já valia antes por não haver
  target neste loop; continua valendo com investigador/judge locais porque
  o loop de tool-calling interativo exige o guardrail vivo respondendo
  `get_jlens_readout` no meio da conversa do investigador — carregar tudo
  junto (em vez de separar guardrail+investigador de um lado e judge de
  outro) só deixou de ser um problema de memória depois que tiramos o vLLM
  da equação: carregar via `transformers.from_pretrained` aloca exatamente
  o que os pesos precisam, sem a reserva antecipada de KV-cache que um
  servidor vLLM faria por processo. Ver orçamento de VRAM na seção
  dedicada abaixo.

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

### 3. `audit_agent.py` — investigador (DeepSeek) + judge (Groq), via API

*(Nota: esta seção descrevia originalmente um design baseado em Gemini —
implementado depois com um design diferente, é o que está documentado
abaixo e é o que roda de fato hoje no path API/dry-run CPU.)*

- Wrap de `GuardrailLens.readout()`/`readout_multi()` como *tool* no
  formato de function calling OpenAI-compatible: nome `get_jlens_readout`,
  parâmetros `positions: list[int]`, `layers: list[int] | None`
  (`READOUT_TOOL_SCHEMA`). O investigador vê um mapa posição→token
  (`GuardrailLens.token_span`) e escolhe, por julgamento próprio, quais
  posições consultar — não fica preso à posição de decisão (`-1`).
- Loop manual de tool-calling: manda transcript + claim pro investigador
  (**DeepSeek** `deepseek-ai/deepseek-v4-pro`, via endpoint OpenAI-compatible
  da NVIDIA) com `tools=[READOUT_TOOL_SCHEMA]`; a cada function call,
  executa `gl.readout_multi(...)` localmente e devolve o resultado como
  mensagem `tool`; repete até `--max-tool-calls` ou até ele responder com
  veredito final em JSON (evidência citada + claim resolvida sim/não).
  Fallback pra uma leitura fixa na posição `-1` (`_investigate_fixed`) se a
  primeira chamada com `tools` falhar (endpoint não suporta function
  calling pra esse modelo).
- Judge: chamada separada pro **Groq** (`openai/gpt-oss-120b`), sem acesso à
  ferramenta — recebe `(claim, resposta do investigador, gabarito
  esperado)`, devolve `correctness`/`evidence_quality` 0-10 + justificativa.
  Provedor diferente do investigador de propósito (reduz viés de
  auto-avaliação de uma mesma família de modelo se avaliando).
- Carrega as chaves via `python-dotenv` a partir do `.env` da raiz do repo
  (`DEEPSEEK_API_KEY`/`DEEPSSEK_API_KEY` e `GROQ_API_KEY`/`GPT_API_KEY`).

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

*(Implementado com `openai` + `groq` + `python-dotenv`, não
`google-generativeai` — reflexo do mesmo design atualizado da seção 3.)*

- `openai` (cliente do investigador, endpoint DeepSeek/NVIDIA), `groq`
  (cliente do judge), `python-dotenv` em `guardrail_eval/requirements.txt`
  — já feito. Quando o path local (seção 7) for implementado, não precisa
  de dependência nova: `transformers` já está em `requirements.txt`.
- **`.env` no `.gitignore` da raiz** — já corrigido; guarda
  `DEEPSEEK_API_KEY`/`DEEPSSEK_API_KEY` e `GROQ_API_KEY`/`GPT_API_KEY`.

### 6. `target_model.py` (touch mínimo, preparo futuro)

Adicionar o mesmo parâmetro `device` que `GuardrailLens` recebe, e mover
`encoded` (dict do `apply_chat_template`) pro device antes de
`self.hf.generate(**encoded, ...)` — hoje ele fica implicitamente em CPU
mesmo que o modelo esteja em GPU. Não integrado ao `run_audit_pipeline.py`
agora (target fora de escopo), mas deixa `target_model.py` pronto pra
quando o target voltar a entrar no pipeline.

### 7. Novo (planejado, não implementado ainda): modelos abertos locais
   pro investigador/judge — só na run do RunPod

Motivação: os testes via API (DeepSeek/NVIDIA + Groq) estavam lentos demais
pra iterar. vLLM resolveria a velocidade mas traria riscos próprios — não
determinismo por batching contínuo mesmo em `temperature=0`, incerteza
sobre se `tools`/`response_format` são de fato respeitados por cada
modelo+versão, e contenção de VRAM entre processos vLLM concorrentes.
Decisão: **sem vLLM** — investigador e judge passam a rodar como modelos
`transformers` locais, mesmo padrão de carregamento que `GuardrailLens` já
usa, só na run do pod. **O dry-run local em CPU não muda** — continua 100%
via API (seção 3 acima).

Novo módulo `guardrail_eval/local_agent.py`, espelhando a interface pública
de `audit_agent.py` (`investigate(...)`, `judge(...)`) com backend local:

- Carrega investigador e judge via `AutoModelForCausalLM.from_pretrained`
  (mesmo padrão de `GuardrailLens.__init__`), não via cliente de API.
- Tool-calling do investigador: `tokenizer.apply_chat_template(messages,
  tools=[READOUT_TOOL_SCHEMA], add_generation_prompt=True)` — suportado
  nativamente por Qwen2.5-Instruct e Llama-3.1/3.3-Instruct sem precisar de
  servidor. O modelo emite o tool-call como texto (formato
  `<tool_call>{...}</tool_call>` no caso do Qwen); o parse desse bloco é
  manual. A resposta da ferramenta (via o mesmo `readout_fn`/
  `gl.readout_multi` já usado hoje) volta pro histórico como próximo turno;
  repete até o modelo parar de chamar a ferramenta ou bater
  `--max-tool-calls`.
- Veredito final e score do judge: um `generate()` simples + reaproveita
  `_extract_json` (importado de `audit_agent.py`, não duplicado) pra
  extrair o JSON do texto gerado — sem grammar-constrained decoding, então
  o formato depende só do prompt (menos garantido que um parser de
  servidor; vale checar manualmente nas primeiras linhas do primeiro run).
- Modelos recomendados como default (famílias diferentes, preservando a
  lógica de "cross-provider" contra viés de auto-avaliação): investigador
  `Qwen/Qwen2.5-14B-Instruct-AWQ` (4-bit), judge
  `meta-llama/Llama-3.1-8B-Instruct` (bf16). Documentar como chute inicial
  a calibrar no primeiro run, no mesmo espírito de `--max-tool-calls`.
- Novos flags em `run_audit_pipeline.py`, **independentes um do outro**:
  `--investigator-backend {api,local}` e `--judge-backend {api,local}`,
  cada um default `api` (preserva o dry-run em CPU sem nenhuma mudança de
  comportamento). Permite testar combinações mistas no pod (ex.
  investigador local + judge via API) antes de comprometer com os dois
  locais. `--investigator-model`/`--judge-model` passam a aceitar tanto
  nomes de modelo de API quanto repo IDs do HF, dependendo do backend
  escolhido pra cada papel.
- **Sem separação de fases**: guardrail + investigador + judge residentes
  juntos o loop inteiro, exatamente como o loop único de hoje
  (`run_audit_for_attack`: classify → readout → investigate → judge por
  linha) — só troca de onde vêm `investigate`/`judge` conforme os dois
  flags acima. `--resume` não muda: mesmo checkpoint único de hoje, baseado
  em `pool_index` já presente em `audit_readouts_<attack>.jsonl`.

**Orçamento de VRAM** (pod de 48GB, os três residentes juntos o tempo
todo): guardrail `gemma-3-4b-it` bf16 (~8GB — não quantizar, é o objeto de
medição do experimento) + investigador `Qwen2.5-14B-Instruct-AWQ` (~9GB,
4-bit) + judge `Llama-3.1-8B-Instruct` bf16 (~16GB) = **~33GB**, deixando
~15GB pra KV-cache/ativações — cabe confortável. Se quantizar o judge
também (AWQ, ~5.5GB), total cai pra **~22.5GB**, com ~25GB de folga — mais
margem de segurança, ao custo de mais uma perda de qualidade no judge. A
primeira opção (só investigador quantizado) é o default recomendado.

### 8. Novo (planejado): metadata de proveniência

Hoje nenhum artefato registra qual modelo gerou cada linha — sem isso,
comparar uma run local (open-weight) com uma futura via API não é
rastreável. Independente do backend:

- `guardrail_model` em cada linha de `audit_readouts_<attack>.jsonl`.
- `investigator_model` + `judge_model` (+ `investigator_backend`/
  `judge_backend`: `"api"`/`"local"`) em cada linha de
  `audit_scores_<attack>.jsonl`.
- `audit_run_meta_<attack>.json`, escrito uma vez no início da run:
  `vars(args)` completo (modelo, dtype, device, layers, top-k,
  max-tool-calls, backends, etc.) — reprodutibilidade total da
  configuração, não só o nome do modelo.

## Setup do RunPod (execução manual)

Execução planejada via **VSCode Remote-SSH** direto no pod (editor +
terminal integrado rodando no host remoto).

1. Pod GPU real desta run: **48GB VRAM, 50GB RAM, 9 vCPU**, imagem
   `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8.1, torch
   2.8.0 de fábrica, Ubuntu 24.04 — template diferente do usado numa sessão
   anterior, que era `2.4.0-...-cuda12.4.1-...-ubuntu22.04`; ver
   "Dependências do pod" abaixo, que agora deriva a versão de CUDA
   dinamicamente em vez de assumir uma fixa, então serve pra qualquer
   template). 48GB de VRAM dá folga
   confortável pro guardrail (~8GB `gemma-3-4b-it` bf16) **e** pro
   investigador+judge locais rodando juntos (ver orçamento de VRAM na
   seção "Modelos abertos locais pro investigador/judge" acima — ~33GB de
   pesos no total, ~15GB de sobra pra KV-cache/ativações). Confirmar na
   criação do pod que **SSH Terminal Access** está habilitado (alguns
   templates só dão terminal web, sem `ssh` de linha de comando — não
   serve pro Remote-SSH) e que sua chave pública SSH está cadastrada na
   conta RunPod.
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
   3 estágios recomendado antes de comprometer com o N final), **trocando
   o guardrail para gemma-3-4b-it** (via `--guardrail-model`; o preset já
   resolve a lente e os quirks de carregamento) **e o investigador/judge
   pro backend local** (`--investigator-backend local --judge-backend
   local` — ver "Modelos abertos locais pro investigador/judge" acima;
   omitir os dois flags mantém o path API/DeepSeek+Groq de sempre):
   ```
   python run_audit_pipeline.py --guardrail-model google/gemma-3-4b-it \
       --device cuda --dtype bfloat16 --attack both \
       --investigator-backend local --judge-backend local \
       --n-malign 15 --n-benign 15 --resume --verbose
   ```
   e inspecionar `results_audit/audit_scores_baseline.jsonl`,
   `results_audit/audit_scores_baseline_wrapping.jsonl` e
   `results_audit/audit_summary_combined.csv`.
6. Derrubar o pod depois de validar (billing por minuto) — encerrar a
   sessão Remote-SSH e terminar o pod pelo dashboard do RunPod.

### Dependências do pod (a causa de uma sessão de debug anterior)

Numa sessão anterior, um template mais antigo (`2.4.0-...-cuda12.4.1-...`)
trazia torch 2.4.0, mas `transformers>=5.5` (exigido pelo `jlens`) —
especificamente a 5.14.1 — importa `torch.distributed.tensor.DTensor`, que
não existia no torch 2.4.0 (`ImportError`). Corrigir isso rodando `pip
install --upgrade torch` sozinho quebrou `torchvision`/`torchaudio` em
cadeia (ABI incompatível: `operator torchvision::nms does not exist`,
depois `undefined symbol` no `libtorchaudio.so`) — cada fix isolado de uma
lib quebrava a próxima.

**`guardrail_eval/setup_pod.sh` automatiza a correção certa, de forma
adaptável ao template** (o template real muda entre sessões — já rodamos em
`2.4.0/cu124` e agora `2.8.0/cu128`, ver spec do pod acima):
- Testa se `DTensor`/`transformers` importam (o problema real). Não gate em
  `torchvision`/`torchaudio` — nenhum dos dois é dependência de
  `jlens`/`guardrail_eval` (confirmar em `requirements.txt`/
  `pyproject.toml`), então um template que simplesmente não os inclui não
  deve disparar reinstalação nenhuma.
- Se o teste falhar, reinstala torch+torchvision+torchaudio **juntos, de um
  índice CUDA derivado dinamicamente de `torch.version.cuda`** (ex.: `12.8`
  → `.../whl/cu128`) — nunca de um índice fixo. Isso é o que evita o script
  rebaixar/desencontrar um template mais novo (como o `cu1281` atual) pra um
  build de CUDA mais antigo só porque foi escrito pensando no template
  anterior.
- Se ainda assim falhar depois da reinstalação casada, o script sai com
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

### Investigador multi-posição + tool-calling interativo (path API)

*(Esta seção cobre `--investigator-backend api`/`--judge-backend api` — o
path usado pelo dry-run local em CPU e, no pod, disponível pra testes
mistos. Pro path local, ver a seção seguinte.)*

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

### Investigador/judge locais (a fazer no pod, path `--*-backend local`)

Caminho de código novo (`local_agent.py`, ver seção "Modelos abertos locais
pro investigador/judge" acima) — sem equivalente já validado, diferente do
path API que só está pendente de rodar numa GPU de verdade.

**Verificar no PRIMEIRO run local no pod:**
- Que o parsing do tool-call (bloco de texto emitido pelo modelo, não um
  campo estruturado de servidor) está de fato extraindo `positions`/
  `layers` corretos — sem grammar-constrained decoding, isso depende só do
  prompt, então checar manualmente que não está falhando silenciosamente
  (linhas caindo num fallback sem motivo real).
- Que o parsing do JSON final (`_extract_json`, reaproveitado de
  `audit_agent.py`) está extraindo `verdict`/`evidence` corretamente do
  texto gerado pelo investigador/judge locais.
- Que os três modelos (guardrail + investigador AWQ + judge) cabem
  simultaneamente nos 48GB sem OOM — conferir uso de VRAM (`nvidia-smi`)
  durante uma linha completa, não só no carregamento.
- Rodar pelo menos uma combinação mista (`--investigator-backend local
  --judge-backend api`, ou o inverso) antes de comprometer com os dois
  locais — isola se um problema é do investigador, do judge, ou da
  interação entre os dois.
- Que os campos de proveniência (`guardrail_model`, `investigator_model`,
  `judge_model`, `investigator_backend`/`judge_backend`) aparecem em toda
  linha nova.

### Run principal

Escala anterior (60 prompts: 15 malign + 15 benign por ataque) foi reduzida
de propósito por preocupação de free tier de API — não é o corpus completo
(200 malign + 30 benign por ataque = 460 prompts). Com investigador/judge
locais, o teto deixa de ser rate-limit de API e passa a ser tempo de
parede/billing do pod — vale reconsiderar o alvo.

**Orçamento de tempo/GPU esperado** (path local): sem chamadas de API, não
há mais rate-limit/429 a gerenciar (isso só se aplica quando algum dos dois
backends está em `api`) — o custo agora é puramente tempo de GPU, cobrado
por minuto do pod. Sem uma medição real ainda, não dá pra estimar
linha/minuto com confiança — daí o rollout abaixo incluir um estágio
dedicado só a medir isso antes de decidir o N final.

**Rollout recomendado (3 estágios):**
1. **Smoke do código novo**: `--investigator-backend local --judge-backend
   local --attack baseline --n-malign 2 --n-benign 2` — confirma que o
   path local roda fim-a-fim (parsing de tool-call, parsing de JSON final,
   campos de proveniência, `--resume` ainda funciona normalmente) antes de
   gastar tempo de GPU em escala. Rodar também uma combinação mista (ver
   seção anterior) pra isolar problemas por papel.
2. **Checkpoint cronometrado, na escala anterior**: `--attack both
   --n-malign 15 --n-benign 15 --investigator-backend local
   --judge-backend local --resume --verbose` — medir tempo de parede total
   e extrapolar linha/minuto. `--resume` continua disponível caso o pod
   caia no meio (mesmo checkpoint único de sempre, baseado em `pool_index`
   já em `audit_readouts_<attack>.jsonl`).
3. **Decidir o N final com base na taxa medida no estágio 2.** Exemplo de
   raciocínio: se as 60 linhas do estágio 2 levaram X minutos, o corpus
   completo (200 malign + 30 benign por ataque × 2 ataques = 460 linhas)
   leva ~7.7×X — comparar contra o orçamento de tempo/custo do pod antes de
   comprometer. Se couber, rodar `--n-malign 200 --n-benign 30` (as 230
   seeds reais de `seed_pool.csv`, não mais os 15+15 reduzidos); se não,
   escolher um N intermediário ou dividir em múltiplas sessões de pod
   apoiado em `--resume`.

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
  `audit_readouts_<attack>.jsonl` — vale tanto pro path API quanto local
  (mesmo checkpoint único, não há checkpoint duplo a testar).
- Se aparecerem 429 recorrentes (só relevante quando algum backend está em
  `api`), ajustar `--api-pacing-seconds` e/ou `--max-tool-calls` antes de
  tentar de novo — não simplesmente re-rodar esperando sorte.
- **Path local**: confirmar que o parsing de tool-call/JSON não está caindo
  em fallback silencioso sem motivo, que os três modelos cabem juntos em
  VRAM sem OOM durante uma linha completa (não só no carregamento), que os
  campos de proveniência (`guardrail_model`/`investigator_model`/
  `judge_model`/`*_backend`) estão presentes em toda linha nova, e que ao
  menos uma combinação mista de backends (`local`+`api`) foi testada.
- Conferir que `guardrail_eval/data/`, `jlens/`, `tests/`, `data/` e
  `harmbench.csv` da raiz permanecem intocados (mesma restrição de sempre
  do sub-projeto).
