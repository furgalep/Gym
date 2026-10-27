# NOOA Agent

Runs any [NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents) agent as a NeMo Gym harness, inside the task
sandbox, for evaluation and for RL.

## How it works

```
Gym agent server (this directory)          task sandbox
───────────────────────────────            ──────────────────────────────────────────────
seed:  stage the NOOA runtime archive  ──► /tmp/nemo-gym-nooa-runtime/<digest>/python
       open a SandboxSession
turn:  write input.json                ──► python -I -m nooa_bench.gym_worker in.json out.json
       (task, agent, turn_method,            constructs <agent>(llm=<Gym model client>, **agent_kwargs)
        tools, model URL, ...)               binds Gym tools into the NOOA interface
                                             await agent.<turn_method>(task)
                                             every model call ──► Gym model server (rollout-prefixed URL)
       read result.json               ◄──  observations, terminal response id, stop reason
close: observations to Gym, release the sandbox
```

The server never imports NOOA. The boundary is the worker's JSON payload and result (`Episode` in `app.py`); the
only vocabulary both sides share is the Responses API. Gym's model server holds the authoritative capture of every
model call, so the harness returns observations only and never restates model calls or builds a trajectory.

## Any NOOA agent, by configuration

```yaml
agent: nooa_bench.gym:ShellAgent   # module:Class
turn_method: solve                 # awaited with the task text
agent_kwargs: {}                   # constructor arguments beyond llm=
```

The worker constructs the class with `llm=` (NOOA's Responses client pointed at Gym) and `agent_kwargs`. A
constructor that names `run_command`, `workdir` or `working_dir` also receives that; nothing else is guessed.

Gym's function tools are translated into the NOOA interface by `nooa_bench.gym.bind_gym_tools`: a Skill of
documented async callables on the agent (`self.gym`, discoverable through `doc(self)`) for CodeAct agents, plus
`Tool` objects on `agent.extra_tools` for agents that drive a tool-calling loop. `tool_binding:
module:callable(agent, tools)` replaces that translation for a specific harness.

## Runtime

Nothing is resolved or installed inside the sandbox. Build the runtime once per libc from the NOOA revision pinned
in `requirements.txt`, then point the server at the archives (or set `NOOA_GYM_RUNTIME_TARBALL` and
`NOOA_GYM_RUNTIME_TARBALL_MUSL`):

```bash
# <nooa> is a checkout of labs-OO-Agents at the pinned revision; local directories are built into wheels.
python -m responses_api_agents.nooa_agent.runtime build --output nooa-runtime-x86_64.tar.gz --arch x86_64 \
    <nooa> <nooa>/packages/nooa-cli <nooa>/packages/nooa-bench "jsonschema>=4.25.1,<5"
python -m responses_api_agents.nooa_agent.runtime build --output nooa-runtime-x86_64-musl.tar.gz \
    --target x86_64-unknown-linux-musl <nooa> <nooa>/packages/nooa-cli <nooa>/packages/nooa-bench "jsonschema>=4.25.1,<5"
```

Set `UV_CONSTRAINT=<constraints file>` while building to pin third-party versions across rebuilds.

Staging probes the image's libc, uploads the matching archive, extracts it under a digest-named directory and runs
one import check. Local sandbox providers can instead set `runtime_python` to an interpreter that already has NOOA.

## Outcomes

| Stop reason | Response status | Graded |
|---|---|---|
| `completed` | `completed` | yes |
| `policy_budget_exceeded`, `max_output_tokens_exceeded` | `incomplete` | yes |
| `timeout_exceeded` (supervisor SIGTERM at the agent budget; worker checkpointed) | `incomplete` | yes, the sandbox state |
| anything else | `failed` | no |

A `timeout_exceeded` result the supervisor does not vouch for is a failure, not a budget stop. The response
output holds only the policy's own items; `metadata.terminal_response_id` names the root agent's last completed
model response so Gym's token-id capture attributes the main chain without a synthetic message.

## Artifacts

`artifacts: [atif, trace_jsonl]` turns on NOOA's own trajectory export and span files inside the sandbox;
with `artifacts_output` set, each episode's artifacts are collected into `<artifacts_output>/<rollout_id>.tar.gz`.

## Tests

```bash
uv pip install -r responses_api_agents/nooa_agent/requirements.txt --override responses_api_agents/nooa_agent/overrides.txt
pytest responses_api_agents/nooa_agent/tests
```

`test_preflight.py` drives a real NOOA harness turn through Gym's strict request converter, which catches
request-shape failures that otherwise appear only on a cluster. `test_budget_stop.py` runs the real process
supervisor against the real worker.
