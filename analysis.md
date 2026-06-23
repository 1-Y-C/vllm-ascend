# Qwen3.5/3.6 full inference pipeline trace on vllm-ascend

## Overview

This analysis traces every step from `vllm serve Qwen/Qwen3.5-0.8B` to token generation with the GDN fused kernel.

Key files:
- `/workspace/user_data/vllm/vllm/entrypoints/cli/serve.py` — CLI entry point
- `/workspace/user_data/vllm/vllm/entrypoints/openai/api_server.py` — HTTP server
- `/workspace/user_data/vllm/vllm/v1/engine/async_llm.py` — AsyncLLM engine
- `/workspace/user_data/vllm/vllm/v1/engine/core.py` — EngineCore (process loop)
- `/workspace/user_data/vllm/vllm/v1/worker/gpu_worker.py` — Worker (model loading)
- `/workspace/user_data/vllm/vllm/v1/worker/gpu_model_runner.py` — Model runner (forward)
- `/workspace/user_data/vllm/vllm/model_executor/models/qwen3_5.py` — Qwen3.5 model
- `/workspace/user_data/vllm-ascend/vllm_ascend/ops/gdn.py` — Ascend GDN core
- `/workspace/user_data/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py` — Monkey-patching

---

# Part 1: Server Startup Flow

## Step 1.1: CLI Entry — `vllm serve Qwen/Qwen3.5-0.8B`

**File:** `vllm/entrypoints/cli/serve.py`

The `vllm` CLI is defined in `setup.py` via `console_scripts`. The `serve` subcommand maps to `ServeSubcommand`.

`vllm serve Qwen/Qwen3.5-0.8B` calls `ServeSubcommand.cmd(args)` (line 50):

```python
@staticmethod
def cmd(args: argparse.Namespace) -> None:
    # args.model = "Qwen/Qwen3.5-0.8B"
    # args.grpc defaults to False
    # args.headless defaults to False
    # args.api_server_count defaults to None → set per LB mode

    # For defaults (no data parallel flags):
    # args.api_server_count = None → defaulted to args.data_parallel_size (= 1 for single GPU)
    # is_multi_port = is_external_lb = is_hybrid_lb = False
    # Falls through to single API server branch (line 148):
    uvloop.run(run_server(args))
```

**Decision chain at line 55-148:**
1. If `args.grpc`: run gRPC server → return (NOT taken)
2. If `args.headless`: run without API server → return (NOT taken)
3. Determine LB mode from DP flags (NOT taken for single GPU)
4. If multi-port LB: `run_dp_supervisor(args)` (NOT taken)
5. If `args.api_server_count < 1`: `run_headless(args)` (NOT taken)
6. If `args.api_server_count > 1` or Rust frontend: `run_multi_api_server(args)` (NOT taken)
7. **Single API server** (line 148): `uvloop.run(run_server(args))` ← **TAKEN**

## Step 1.2: Server Bootstrap — `run_server` and `run_server_worker`

**File:** `vllm/entrypoints/openai/api_server.py`

### `run_server` (line 665):
```python
async def run_server(args, **uvicorn_kwargs) -> None:
    decorate_logs("APIServer", skip_if_decorated=True)
    # Register SIGTERM handler
    signal.signal(signal.SIGTERM, _interrupt_init)
    listen_address, sock = setup_server(args)  # Creates socket, binds to host:port
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
```

### `run_server_worker` (line 681):
```python
async def run_server_worker(listen_address, sock, args, ...):
    # 1) Create engine client (context manager)
    async with build_async_engine_client(args) as engine_client:
        # 2) Build FastAPI app + init state + start uvicorn
        shutdown_task = await build_and_serve(engine_client, listen_address, sock, args)
    # 3) Wait for shutdown
    await shutdown_task
    sock.close()
```

## Step 1.3: Engine Client Creation — `build_async_engine_client`

**File:** `vllm/entrypoints/openai/api_server.py`, line 78

```python
async def build_async_engine_client(args, ...):
    engine_args = AsyncEngineArgs.from_cli_args(args)
    async with build_async_engine_client_from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    ) as engine:
        yield engine
```

### `build_async_engine_client_from_engine_args` (line 109):
```python
# 1) Create VllmConfig
vllm_config = engine_args.create_engine_config(usage_context=usage_context)
# 2) Create AsyncLLM
async_llm = AsyncLLM.from_vllm_config(
    vllm_config=vllm_config,
    usage_context=usage_context,
    enable_log_requests=engine_args.enable_log_requests,
    ...
)
# 3) Reset multimodal cache
await async_llm.reset_mm_cache()
yield async_llm
```

## Step 1.4: AsyncLLM Initialization

**File:** `vllm/v1/engine/async_llm.py`, line 70

### `AsyncLLM.__init__`:
```python
def __init__(self, vllm_config, executor_class, ...):
    # 1) Store configs
    self.vllm_config = vllm_config
    self.model_config = vllm_config.model_config

    # 2) Create renderer (tokenizer)
    self.renderer = renderer_from_config(self.vllm_config)

    # 3) Create input processor (converts EngineInput → EngineCoreRequest)
    self.input_processor = InputProcessor(self.vllm_config, renderer)

    # 4) Create output processor (converts EngineCoreOutputs → RequestOutput)
    self.output_processor = OutputProcessor(renderer.tokenizer, ...)

    # 5) Create EngineCore (runs in separate process via MPClient)
    self.engine_core = EngineCoreClient.make_async_mp_client(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=self.log_stats,
        ...
    )

    # 6) Create stat loggers
    if self.log_stats:
        self.logger_manager = StatLoggerManager(...)

    # 7) Start output handler (background asyncio task)
    self._run_output_handler()  # Polls engine_core output queue → pushes to per-request queues
```

### EngineCoreClient.make_async_mp_client (line 108-153 in core_client.py):
Creates an `MPClient` which spawns an `EngineCoreProc` (a child process running EngineCore).

## Step 1.5: EngineCoreProc Startup (Child Process)

**File:** `vllm/v1/engine/core.py`

### `EngineCoreProc.__init__` → `EngineCore.__init__` (line 94):
```python
class EngineCore:
    def __init__(self, vllm_config, executor_class, log_stats, ...):
        # 1) Create executor (e.g., MultiprocExecutor for multi-GPU, 
        #    or direct worker for single GPU)
        self.model_executor = executor_class(vllm_config)

        # 2) Initialize KV caches + update CacheConfig after memory profiling
        kv_cache_config = self._initialize_kv_caches(vllm_config)

        # 3) Create scheduler
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
        self.scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            ...
        )
```

### Key: Executor == WorkerProc for single GPU
For single-GPU Ascend, `executor_class` is the Ascend executor. The executor creates a `WorkerProc` which starts a child process running `Worker.init_device()` then `Worker.load_model()`.

## Step 1.6: Worker.init_device (Device + Dist Init)

**File:** `vllm/v1/worker/gpu_worker.py`, line 236

```python
def init_device(self):
    # 1) Set device (NPU 0 for single GPU)
    self.device = torch.device(f"cuda:{self.local_rank}")  # NPU maps to cuda:0
    torch.accelerator.set_device_index(self.device)

    # 2) Initialize distributed environment (torch.distributed)
    init_worker_distributed_environment(
        self.vllm_config, self.rank, self.distributed_init_method, self.local_rank
    )

    # 3) Take memory snapshot for KV cache sizing
    self.init_snapshot = MemorySnapshot(device=self.device)
    self.requested_memory = request_memory(init_snapshot, self.cache_config)

    # 4) Initialize workspace manager
    init_workspace_manager(self.device, num_ubatches=1 or 2)

    # 5) Construct GPUModelRunner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    self.model_runner = GPUModelRunner(self.vllm_config, self.device)
```

## Step 1.7: Model Loading — `Worker.load_model()`

**File:** `vllm/v1/worker/gpu_model_runner.py`, line 5032

```python
def load_model(self, load_dummy_weights=False):
    # 1) Get model loader based on load_config (default: default_model_loader)
    model_loader = get_model_loader(self.load_config)

    # 2) Load the actual model (creates Qwen3_5ForCausalLM)
    self.model = model_loader.load_model(
        vllm_config=self.vllm_config,
        model_config=self.model_config
    )

    # 3) Post-load: set up EPLB (if MoE), LoRA, etc.
    #    Also wrap with CUDAGraphWrapper if FULL cudagraph mode
```

### Model Architecture (Qwen3.5-0.8B):
From HuggingFace config.json:
- `num_hidden_layers`: determined by config (e.g., 28 for 0.8B)
- `layer_types`: list specifying which layers are "linear_attention" (GDN) and which are "full_attention"
- Typically: layers 0, 2, 4, ... are GDN (linear_attention), others are full_attention

### Model instantiation chain:
1. `Qwen3_5ForCausalLM.__init__` → creates `Qwen3_5Model`
2. `Qwen3_5Model.__init__` → iterates `config.layer_types`, creates `Qwen3_5DecoderLayer` per layer
3. For `"linear_attention"` layers: `QwenGatedDeltaNetAttention(...)` ← KEY
4. For `"full_attention"` layers: `Qwen3NextAttention(...)`

## Step 1.8: Ascend Monkey-Patching

**File:** `/workspace/user_data/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py`

This file is imported during worker startup (via plugin loading). It monkey-patches the CUDA classes with Ascend implementations:

```python
# Line 22: Target class
_GDN_PATCH_TARGET = QwenGatedDeltaNetAttention  # The CUDA GDN class

# Lines 143-157: Replace methods
_GDN_PATCH_TARGET.forward = AscendGatedDeltaNetAttention.forward
_GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention._forward_core
_GDN_PATCH_TARGET._warmup_prefill_kernels = AscendGatedDeltaNetAttention._warmup_prefill_kernels
_GDN_PATCH_TARGET._split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp
_GDN_PATCH_TARGET.get_state_shape = AscendGatedDeltaNetAttention.get_state_shape
_GDN_PATCH_TARGET.get_attn_backend = AscendGatedDeltaNetAttention.get_attn_backend

# Lines 143-144: Also patch decoder layer and attention
Qwen3_5DecoderLayer.forward = AscendQwen3_5DecoderLayer.forward
Qwen3NextAttention.forward = AscendQwen3NextAttention.forward
```

After patching, when `self.linear_attn(hidden_states, output)` is called, it runs `AscendGatedDeltaNetAttention.forward()` from `gdn.py`.

## Step 1.9: Graph Capture (CUDAGraph)

After model loading, `GPUModelRunner` wraps the model with `CUDAGraphWrapper` (for FULL mode) and performs a warmup forward pass to capture the graph.

### For FULL cudagraph mode:
- A single decode step is traced and replayed for every subsequent decode step
- The model forward → all layers → all ops are recorded as a graph
- During capture, `_EXTRA_CTX.capturing = False` initially, but the ACL graph framework sets it to `True` during the actual graph capture phase

### For PIECEWISE mode:
- Only certain ops are compiled into graphs
- `qwen_gdn_attention_core` is a "splitting op" (defined in `vllm/config/compilation.py`)
- This means the graph is cut at this op, allowing the GDN attention core to be executed outside of the graph

## Step 1.10: Server Start — `build_and_serve`

**File:** `vllm/entrypoints/openai/api_server.py`, line 572

```python
async def build_and_serve(engine_client, listen_address, sock, args, **uvicorn_kwargs):
    # 1) Get supported tasks from engine
    supported_tasks = await engine_client.get_supported_tasks()

    # 2) Build FastAPI app with endpoints
    app = build_app(args, supported_tasks, model_config)

    # 3) Initialize app state (creates serving handlers)
    await init_app_state(engine_client, app.state, args, supported_tasks)

    # 4) Start uvicorn HTTP server
    return await serve_http(app, sock=sock, host=args.host, port=args.port, ...)
```

### Endpoints registered:
- `POST /v1/completions` → `create_completion` (completion/api_router.py:46)
- `POST /v1/chat/completions` → `create_chat_completion`
- etc.

The server is now **running and ready to accept requests**.

---

# Part 2: Request Handling Flow

## Step 2.1: HTTP Request Arrives

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.5-0.8B", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0}'
```

The request arrives at FastAPI → uvicorn dispatches to the route handler.

## Step 2.2: Request Validation + Parsing

**File:** `vllm/entrypoints/openai/completion/api_router.py`, line 46

```python
@router.post("/v1/completions", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    # 1) Get OpenAIServingCompletion handler from app state
    handler = completion(raw_request)  # = raw_request.app.state.openai_serving_completion

    # 2) Delegate to handler
    generator = await handler.create_completion(request, raw_request)

    # 3) Return based on type:
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=...)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())
    # Streaming:
    return StreamingResponse(content=generator, media_type="text/event-stream")
```

For our request (`max_tokens=7`, `temperature=0`), this is a **non-streaming** completion.

## Step 2.3: OpenAIServingCompletion.create_completion

**File:** `vllm/entrypoints/openai/completion/serving.py`, line 110

```python
async def create_completion(self, request, raw_request):
    return await self._create_completion(request, raw_request)
```

### `_create_completion` (line 128):
```python
async def _create_completion(self, request, raw_request):
    # 1) Validate + preprocess request → list[EngineInput]
    result = await self.render_completion_request(request)
    engine_inputs = result  # EngineInput(prompt_token_ids=[...])

    # 2) Create request ID
    request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"

    # 3) Determine max_tokens, create SamplingParams
    max_tokens = get_max_tokens(max_model_len, request.max_tokens, prompt_len, ...)
    sampling_params = request.to_sampling_params(max_tokens, self.default_sampling_params)
    # sampling_params: max_tokens=7, temperature=0

    # 4) Submit to engine
    generator = self.engine_client.generate(
        engine_input,      # EngineInput with token_ids=[...]
        sampling_params,    # SamplingParams(max_tokens=7, temperature=0)
        request_id_item,    # "cmpl-xxx-0"
    )
    generators.append(generator)

    # 5) Merge multiple generators (for n>1) — here n=1, so one generator
    result_generator = merge_async_iterators(*generators)

    # 6) Non-streaming: collect all outputs
    async for i, res in result_generator:
        final_res_batch[i] = res

    # 7) Format response
    response = self.request_output_to_completion_response(
        final_res_batch_checked, request, request_id, ...
    )
    return response
```

## Step 2.4: AsyncLLM.generate — Request Submission

**File:** `vllm/v1/engine/async_llm.py`, line 524

```python
async def generate(self, prompt, sampling_params, request_id, ...):
    # 1) Process input → EngineCoreRequest
    queue = await self.add_request(
        request_id, prompt, sampling_params, ...
    )
    # Returns a RequestOutputCollector (async queue)

    # 2) Poll the queue for outputs
    finished = False
    while not finished:
        out = queue.get_nowait() or await queue.get()
        finished = out.finished
        if out is not STREAM_FINISHED:
            yield out  # Yield RequestOutput to caller
```

### `add_request` (line 280):
```python
async def add_request(self, request_id, prompt, params, ...):
    # 1) Process input
    request = self.input_processor.process_inputs(
        request_id, prompt, params, ...
    )
    # Converts EngineInput → EngineCoreRequest (contains token_ids, sampling_params, etc.)

    # 2) Create output collector
    queue = RequestOutputCollector(params.output_kind, request.request_id)

    # 3) Send to engine core (separate process via IPC)
    await self._add_request(request, prompt_text, None, 0, queue)
    return queue
```

### `_add_request` (line 400):
```python
async def _add_request(self, request, prompt, parent_req, index, queue):
    # 1) Register with output processor (this process)
    self.output_processor.add_request(request, prompt, parent_req, index, queue)

    # 2) Send to EngineCore (separate process)
    await self.engine_core.add_request_async(request)
```

## Step 2.5: EngineCore — Request Processing in Background Process

**File:** `vllm/v1/engine/core.py`

The EngineCore runs in a separate process. The `engine_core.add_request_async(request)` sends the `EngineCoreRequest` via a shared message queue to the EngineCore process.

### EngineCore.run_busy_loop (line 1193):
```python
def run_busy_loop(self):
    while self._handle_shutdown():
        # 1) Poll input queue for new requests/aborts
        self._process_input_queue()        # Reads from input_queue, calls _handle_client_request
        # 2) Schedule + execute one step
        self._process_engine_step()        # Calls self.step_fn()
```

### `_handle_client_request`:
Converts `EngineCoreRequest` → `Request` and adds to scheduler:
```python
request = Request.make_request(engine_core_request)
self.scheduler.add_request(request)
```

### `step_fn()` = `step()` or `step_with_batch_queue()` (line 428):
```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    if not self.scheduler.has_requests():
        return {}, False

    # 1) Schedule: pick which requests to process, build SchedulerOutput
    scheduler_output = self.scheduler.schedule()

    # 2) Execute: send SchedulerOutput to worker, get ModelRunnerOutput
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 3) Get grammar bitmask (for structured output)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    # 4) Wait for execution + sample tokens
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

    # 5) Update scheduler state from output
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

## Step 2.6: Scheduler.schedule() → SchedulerOutput

The scheduler decides:
1. **Which requests** to process this step (active requests)
2. **Prefill vs Decode**: based on whether each request has already started
3. **Batch composition**: which requests can be batched together

For our single request with `max_tokens=7`:
- **First step**: Prefill — process the full prompt "San Francisco is a" in one forward pass (generates 1 token)
- **Steps 2-7**: Decode — process one new token per step (autoregressive generation)

The scheduler produces a `SchedulerOutput` containing:
- `num_scheduled_tokens`: per-request token counts
- `scheduled_spec_decode_tokens`: (None for non-speculative)
- `num_prefills`, `num_decodes`: request counts
- Token IDs, positions, etc.

## Step 2.7: GPUModelRunner.execute_model

**File:** `vllm/v1/worker/gpu_model_runner.py`, line 3955

```python
def execute_model(self, scheduler_output, intermediate_tensors=None):
    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

    # 1) Update persistent batch states (KV cache indices, positions, etc.)
    deferred_state_corrections_fn = self._update_states(scheduler_output)

    if not num_scheduled_tokens:
        return EMPTY_MODEL_RUNNER_OUTPUT  # No work

    # 2) Prepare inputs: token IDs, positions, attention metadata
    logits_indices, spec_decode_metadata = self._prepare_inputs(scheduler_output, ...)

    # 3) Determine cudagraph mode for this batch
    cudagraph_mode, batch_desc, should_ubatch, ... = self._determine_batch_execution_and_padding(
        num_tokens, num_reqs, num_scheduled_tokens_np, ...
    )
    # For prefill: PIECEWISE mode
    # For decode: FULL mode (uses captured graph)

    # 4) Forward pass: model(input_ids, positions, ...)
    #    Either running the model directly or replaying a cudagraph
```

### Model forward call:
```python
# The model is called with:
model_output = self.model(
    input_ids=input_ids,        # shape: (num_tokens,)
    positions=positions,        # shape: (num_tokens,)
    intermediate_tensors=None,   # for PP, None otherwise
    inputs_embeds=None,
)
```

## Step 2.8: Model Forward Pass

### Step 2.8.1: `Qwen3_5ForCausalLM.forward()`

**File:** `vllm/model_executor/models/qwen3_5.py`, line 500

```python
def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
    return hidden_states
```

### Step 2.8.2: `Qwen3_5Model.forward()` (inherited from `Qwen3NextModel`)

**File:** `vllm/model_executor/models/qwen3_next.py`, line 505

```python
def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    # Step A: Embed tokens
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)  # VocabParallelEmbedding
        residual = None

    # Step B: Iterate through all transformer layers
    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

    # Step C: Final layer norm
    if get_pp_group().is_last_rank:
        hidden_states, _ = self.norm(hidden_states, residual)

    return hidden_states
```

### Step 2.8.3: `Qwen3_5DecoderLayer.forward()` (monkey-patched)

**File:** `vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py`, line 83

```python
def forward(self, hidden_states, residual, positions, **kwargs):
    # Pre-norm with residual
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    # Determine if first layer needs special handling (flash_comm_v1)
    if self.layer_idx == 0 and _EXTRA_CTX.flash_comm_v1_enabled:
        # Special alloc for sequence parallelism
        self_attention_output = torch.empty((n_out, hidden_dim), ...)
    else:
        self_attention_output = torch.empty_like(hidden_states)

    # Dispatch based on layer type
    if self.layer_type == "linear_attention":
        self.linear_attn(hidden_states=hidden_states, output=self_attention_output)
    elif self.layer_type == "full_attention":
        self.self_attn(hidden_states=hidden_states, output=self_attention_output, positions=positions)

    # Post-attention LayerNorm + MLP
    hidden_states = self_attention_output
    # ... layer scale if enabled ...
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    # ... layer scale if enabled ...
    return hidden_states, residual
```

### Step 2.8.4: `AscendGatedDeltaNetAttention.forward()` (GDN layers)

**File:** `vllm-ascend/vllm_ascend/ops/gdn.py`, line 290

```python
def forward(self, hidden_states, output):
    num_tokens = hidden_states.size(0)

    # === Part 1: Input Projection ===
    if hasattr(self, "in_proj_qkv"):
        # CUDA-style projections (not Qwen3.5 path)
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        b, a = ba.chunk(2, dim=-1)  # or _split_ba_for_tp
    else:
        if not self.gqa_interleaved_layout:
            # Standard Qwen3-Next layout: qkvz projection
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            ba, _ = self.in_proj_ba(hidden_states)
            b, a = self._split_ba_for_tp(ba)  # Monkot-patched
        else:
            # Qwen3.5 interleaved layout: fused_qkvzba_split_reshape_cat
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(...)

    # === Part 2: Core Attention (Custom Op) ===
    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype, device=hidden_states.device,
    )

    # This calls the registered custom op
    torch.ops.vllm.qwen_gdn_attention_core(
        mixed_qkv, b, a, core_attn_out, self.prefix, False
    )
    # → Dispatches to self._forward_core() (monkey-patched)

    # === Part 3: Output Projection ===
    # RMSNormGated: norm(core_attn_out, z)
    core_attn_out = self.norm(core_attn_out.reshape(-1, ...), z.reshape(-1, ...))
    core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
    output[:num_tokens], _ = self.out_proj(core_attn_out)
```

## Step 2.9: `_forward_core` — The Core GDN Algorithm

### `qwen_gdn_attention_core()` dispatching (line 1647):
```python
def qwen_gdn_attention_core(qkv_or_qkvz, b_or_ba, a_or_z_out, core_attn_out, fast_kernel, layer_name):
    layer_name = _resolve_layer_name(layer_name)
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]  # Retrieves the GDN module instance
    # Calls monkey-patched _forward_core:
    self._forward_core(mixed_qkv=qkv_or_qkvz, b=b_or_ba, a=a_or_z_out, core_attn_out=core_attn_out)
```

### `AscendGatedDeltaNetAttention._forward_core()` (line 386):

This is the routing hub. It handles ALL paths: prefill, decode (spec/non-spec), and fused/non-fused.

```python
def _forward_core(self, mixed_qkv, b, a, core_attn_out):
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata

    if attn_metadata is None:
        return  # V1 profile run

    # Unpack metadata
    attn_metadata = attn_metadata[self.prefix]
    has_initial_state = attn_metadata.has_initial_state
    spec_query_start_loc = attn_metadata.spec_query_start_loc
    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
    spec_sequence_masks = attn_metadata.spec_sequence_masks
    spec_token_indx = attn_metadata.spec_token_indx
    non_spec_token_indx = attn_metadata.non_spec_token_indx
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
    ssm_state = self.kv_cache[1]  # Mamba SSM state
    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens

    # === DECISION TREE ===
```

#### Decision: Fused Decode Fast Path (line 419-472)

```python
    if (
        attn_metadata.spec_sequence_masks is None      # Not speculative decoding
        and attn_metadata.num_prefills == 0              # Decode only, no prefill
        and attn_metadata.num_decodes > 0                # At least one decode request
        and not _EXTRA_CTX.capturing                     # NOT during graph capture
    ):
        # FUSED DECODE FAST PATH
        # Combines: conv1d + l2norm + gating + recurrence in 2 kernel calls

        # Step 1: Causal Conv1D (update mode: run_mode=1)
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(size(0), size(2))
        non_spec_qsl_host, non_spec_ci_host = get_causal_conv1d_update_host_args(attn_metadata)
        conv_weights_T = conv_weights.transpose(0, 1)
        activation_num = 1 if self.activation else 0

        output_non_spec = torch.empty_like(mixed_qkv)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            output_non_spec, mixed_qkv, conv_weights_T,
            conv_state=self_kv_cache[0],
            bias_opt=self.conv1d.bias,
            query_start_loc_opt=non_spec_qsl_host,
            cache_indices_opt=non_spec_ci_host,
            initial_state_mode_opt=(),
            num_accepted_tokens_opt=[],
            activation_mode=activation_num,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=1,  # Update mode
        )
        mixed_qkv = output_non_spec  # Conv1d output

        # Step 2: Fused Recurrent Gated Delta Rule (7-step fusion)
        scale_val = float(self.head_k_dim ** -0.5)
        ssm_state_fp32 = ssm_state.to(torch.float32)  # bf16→fp32 for kernel
        result = torch.ops._C_ascend.npu_fused_rgdr_packed_decode(
            mixed_qkv, a, b,
            self.A_log, self.dt_bias.to(dtype=torch.float32),
            ssm_state_fp32,
            non_spec_state_indices_tensor[:num_actual_tokens],
            scale_val,
        )
        ssm_state.copy_(ssm_state_fp32)  # Write back state
        core_attn_out[:num_actual_tokens] = result.squeeze(1)
        return  # <-- EARLY RETURN, fused path complete
```

When `_EXTRA_CTX.capturing` is `True` (during graph capture) or during prefill, the code falls through to the non-fused path.

#### Non-Fused Path (starts at line 474):

```python
    # Non-fused path: separate conv1d → l2norm → fused_gdn_gating → recurrence

    # 1. Conv1d (with spec/non-spec branching)
    conv_weights = self.conv1d.weight.view(...)
    if spec_sequence_masks is not None:
        # Speculative decode: split tokens into spec and non-spec
        mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
        mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
    else:
        # Standard decode/prefill: only non-spec
        mixed_qkv_spec = None
        mixed_qkv_non_spec = mixed_qkv

    # Process spec tokens (conv1d update with graph capture support)
    if spec_sequence_masks is not None:
        if _EXTRA_CTX.capturing:
            # ... record graph ops for speculative conv1d ...
        else:
            torch.ops._C_ascend.npu_causal_conv1d_custom(...)

    # Process non-spec tokens
    if attn_metadata.num_prefills > 0:
        # PREFILL: conv1d in full sequence mode (run_mode=0)
        torch.ops._C_ascend.npu_causal_conv1d_custom(..., run_mode=0)
    elif attn_metadata.num_decodes > 0:
        # DECODE: conv1d in update mode (run_mode=1)
        if _EXTRA_CTX.capturing:
            # Record graph ops for conv1d update
        else:
            torch.ops._C_ascend.npu_causal_conv1d_custom(..., run_mode=1)

    # 2. Rearrange mixed_qkv → separate query/key/value tensors
    query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
    query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

    # 3. Gating: compute g = exp(-exp(A_log) * softplus(a + dt_bias))
    #           and beta = sigmoid(b)
    g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
    # Split into spec/non-spec
    if spec_sequence_masks is not None:
        g_spec = g.index_select(1, spec_token_indx)
        g_non_spec = g.index_select(1, non_spec_token_indx)
        # ...

    # 4. Recurrence
    # For spec tokens:
    if spec_sequence_masks is not None:
        query_spec = l2norm_fwd(query_spec)
        key_spec = l2norm_fwd(key_spec)
        core_attn_out_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=query_spec.squeeze(0),
            key=key_spec.squeeze(0),
            value=value_spec.squeeze(0),
            g=g_spec.squeeze(0),
            beta=beta_spec.squeeze(0),
            state=ssm_state,
            scale=key_spec.shape[-1] ** -0.5,
            actual_seq_lengths=...,
            ssm_state_indices=spec_state_indices_tensor.flatten(),
            num_accepted_tokens=...,
        ).unsqueeze(0)

    # For non-spec tokens:
    if attn_metadata.num_prefills > 0:
        # PREFILL: chunked recurrence (full sequences)
        (core_attn_out_non_spec, last_recurrent_state) = chunk_gated_delta_rule(
            q=query_non_spec, k=key_non_spec, v=value_non_spec,
            g=g_non_spec, beta=beta_non_spec,
            initial_state=..., output_final_state=True,
            cu_seqlens=non_spec_query_start_loc,
            ...
        )
        ssm_state[non_spec_state_indices_tensor] = last_recurrent_state...
    elif attn_metadata.num_decodes > 0:
        # DECODE: single-step recurrence
        query_non_spec = l2norm_fwd(query_non_spec)
        key_non_spec = l2norm_fwd(key_non_spec)
        core_attn_out_non_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=query_non_spec.squeeze(0),
            key=key_non_spec.squeeze(0),
            value=value_non_spec.squeeze(0),
            g=g_non_spec.squeeze(0),
            beta=beta_non_spec.squeeze(0),
            state=ssm_state,
            scale=key_non_spec.shape[-1] ** -0.5,
            actual_seq_lengths=...,
            ssm_state_indices=non_spec_state_indices_tensor,
        ).unsqueeze(0)

    # 5. Merge spec + non-spec outputs
    if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
        merged_out = torch.empty(...)
        merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
        merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
    elif spec_sequence_masks is not None:
        core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
    else:
        core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
```

### Key insight about fused vs non-fused in FULL cudagraph:
During FULL cudagraph capture (`_EXTRA_CTX.capturing=True`):
- The fused path check FAILS (line 423)
- Falls through to the non-fused path
- The non-fused conv1d and recurrence ops are **captured into the graph**
- During decode replay: the graph replays the captured non-fused ops
- The fused path code in `_forward_core` is NEVER executed during actual decode in FULL mode

The fused path only works in:
1. `--enforce-eager` mode (no graph capture)
2. PIECEWISE mode (where `qwen_gdn_attention_core` is a splitting op)

## Step 2.10: Output Processing

After the model forward returns:
1. Hidden states → `compute_logits()` → logits tensor
2. `sample_tokens()` → sample next token IDs from logits
3. Scheduler updates block tables, advances request state
4. EngineCore outputs go to the output queue
5. `output_handler` (background async task in AsyncLLM) dequeues outputs
6. Output processor converts `EngineCoreOutput` → `RequestOutput`
7. `RequestOutput` is pushed to the per-request `RequestOutputCollector`
8. The `generate()` coroutine yields each `RequestOutput`
9. Non-streaming: collect all, format as `CompletionResponse`
10. Return JSON response to HTTP client

### For our request (7 tokens):
| Step | Batch Type | What Happens |
|------|-----------|-------------|
| 1 | Prefill | Process "San Francisco is a" (5 tokens), generate token 1 |
| 2 | Decode | Process token 1, generate token 2 |
| ... | ... | ... |
| 7 | Decode | Process token 6, generate token 7 |

## Step 2.11: GDN Attention Mathematical Details (for reference)

For each decode step, the fused kernel (`npu_fused_rgdr_packed_decode`) performs:
1. **Unpack QKV**: `mixed_qkv` → `Q`, `K`, `V` (shape: [B, heads, dim])
2. **L2NORM Q, K**: Q = Q / ||Q||, K = K / ||K||
3. **Scale Q**: Q = Q * scale (where scale = head_k_dim^(-0.5))
4. **Gate**: gate = exp(-exp(A_log) * softplus(a + dt_bias))
5. **Beta**: beta = sigmoid(b)
6. **Recurrence equation**:
   - delta = V - S @ K  (where S is the state)
   - S = S * gate + K^T * delta * beta
7. **Output**: O = S @ Q

Where S is the per-sequence state matrix of shape [DK, DV].

---

# Summary: Critical Code Paths

```
vllm serve Qwen/Qwen3.5-0.8B
├── ServeSubcommand.cmd()                          [cli/serve.py:50]
│   └── uvloop.run(run_server(args))               [cli/serve.py:148]
│       └── run_server_worker()                    [api_server.py:681]
│           ├── build_async_engine_client()        [api_server.py:78]
│           │   └── AsyncLLM.from_vllm_config()    [async_llm.py:202]
│           │       ├── EngineCoreClient.make_async_mp_client()  [core_client.py:108]
│           │       │   └── EngineCore.__init__()  [core.py:94]
│           │       │       ├── executor_class()   # WorkerProc + Worker
│           │       │       │   ├── init_device()  [gpu_worker.py:236]
│           │       │       │   │   └── GPUModelRunner init
│           │       │       │   └── load_model()   [gpu_model_runner.py:5032]
│           │       │       │       └── Qwen3_5ForCausalLM created
│           │       │       │           └── patch_qwen3_5.py MONKEY-PATCHES
│           │       │       │               GDN forward, _forward_core, etc.
│           │       │       ├── _initialize_kv_caches()
│           │       │       └── Scheduler()
│           │       └── _run_output_handler()
│           └── build_and_serve()                  [api_server.py:572]
│               └── FastAPI + uvicorn → listening on :8080

curl POST /v1/completions {"prompt": "San Francisco is a", "max_tokens": 7}
├── create_completion()                            [completion/api_router.py:46]
│   └── OpenAIServingCompletion.create_completion() [completion/serving.py:110]
│       └── _create_completion()                    [completion/serving.py:128]
│           ├── render_completion_request()         # Tokenize prompt
│           └── engine_client.generate()            [async_llm.py:524]
│               ├── add_request()                   [async_llm.py:280]
│               │   ├── input_processor.process_inputs()
│               │   └── engine_core.add_request_async(request)
│               └── loop: yield RequestOutput
│                   └── [in EngineCore process]:
│                       ├── _process_input_queue()
│                       │   └── scheduler.add_request()
│                       └── _process_engine_step()
│                           ├── scheduler.schedule()
│                           ├── model_executor.execute_model()
│                           │   └── GPUModelRunner.execute_model()  [gpu_model_runner.py:3955]
│                           │       ├── _update_states()
│                           │       ├── _prepare_inputs()
│                           │       ├── _determine_batch_execution_and_padding()
│                           │       └── model.forward(input_ids, positions)
│                           │           └── Qwen3_5Model.forward()
│                           │               └── for each layer:
│                           │                   Qwen3_5DecoderLayer.forward()  [patch]
│                           │                   ├── full_attention:
│                           │                   │   Qwen3NextAttention.forward()  [patch]
│                           │                   └── linear_attention:
│                           │                       AscendGatedDeltaNetAttention.forward()  [patch]
│                           │                       ├── PART 1: Input projection
│                           │                       │   in_proj_qkvz() → mixed_qkv, z
│                           │                       │   in_proj_ba() → b, a
│                           │                       ├── PART 2: Core attention
│                           │                       │   torch.ops.vllm.qwen_gdn_attention_core()
│                           │                       │   → qwen_gdn_attention_core()
│                           │                       │   → self._forward_core()  [Ascend patched]
│                           │                       │   ├── DECODE (not capturing):
│                           │                       │   │   FUSED PATH:
│                           │                       │   │   ├── npu_causal_conv1d_custom()
│                           │                       │   │   └── npu_fused_rgdr_packed_decode()
│                           │                       │   ├── DECODE (capturing) / PREFILL:
│                           │                       │   │   NON-FUSED PATH:
│                           │                       │   │   ├── npu_causal_conv1d_custom()
│                           │                       │   │   ├── rearrange_mixed_qkv()
│                           │                       │   │   ├── fused_gdn_gating()
│                           │                       │   │   ├── l2norm_fwd()
│                           │                       │   │   └── npu_recurrent_gated_delta_rule()
│                           │                       └── PART 3: Output projection
│                           │                           norm() + out_proj()
│                           │       └── sample_tokens()
│                           └── scheduler.update_from_output()
└── Return JSONResponse with generated text
```
