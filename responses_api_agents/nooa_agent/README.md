# NOOA Agent

The NOOA agent adapter provides typed configuration for constructing NOOA agents and invoking them through
asynchronous Python adapters that receive validated NeMo Gym Responses requests.

## Default agent and installation

The reusable `configs/nooa_agent.yaml` configuration runs `nooa_bench.bench_agent:BenchAgent` with
`responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent`. The adapter passes the task prompt to
BenchAgent and preserves its structured result. Set `nooa.agent_class` and `nooa.invocation_adapter` together
when selecting another agent.

Install the agent's requirements from its directory:

```bash
cd responses_api_agents/nooa_agent
uv pip install -r requirements.txt
```

The requirements pin all three NOOA packages: `nooa`, `nooa-cli`, and `nooa-bench`. The CLI supplies executor
support, and the benchmark package supplies the default BenchAgent. Gym's server installation uses this same
requirements file.

## Embedded CodeAct scope

Embedded CodeAct persists for the duration of one invocation. `execute_python` is not computation-only: it can
read and modify files, start processes, use the network, and change other node state. Embedded mode provides no
sandbox protection. Only injected resource methods are Gym/verifier-observable and authoritative; direct CodeAct
side effects belong to NOOA and are not reliably visible to verification or replay. The trajectory distinguishes
the outer `execute_python` call from nested resource calls.

## Sandboxed execution

`nooa.execution_mode: sandboxed` runs the complete NOOA agent in the task sandbox supplied by Resources, as
`python -I -m nooa_bench.gym_worker` on the staged NOOA runtime under Gym's process supervisor. The worker is
NOOA's own in-sandbox episode runner: this server writes a JSON payload (task text, `nooa.agent_class`,
`nooa.turn_method`, `nooa.init_kwargs`, the Gym tools and their grant, the rollout-prefixed model URL) and reads
back an `Episode` (`worker.py`), so the sandbox side never needs Gym code and any NOOA agent class can run.
The worker's model calls go straight to Gym's model server, which holds the authoritative capture; the response
output holds only the policy's own items and `metadata.terminal_response_id` names the root agent's last
completed response for token-id attribution. Gym tools reach the agent through
`nooa_bench.gym.bind_gym_tools` (a Skill of documented callables on `self.gym` plus function tools), or through
the `nooa.tool_binding` callable a harness names. `runtime_python` points at an interpreter that already has
NOOA inside the sandbox and skips runtime preparation (local providers, conformance). Resources owns sandbox
creation and teardown.
Agent and resource sessions use Gym's existing `single_agent_turn` environment. Agent close returns the evidence and final
Resources cookies before verification; Resources closes its sandbox afterward.

`sandbox_runner_timeout_seconds` sets a finite worker ceiling (default: 21420 seconds). Request metadata
`agent_timeout_sec` can shorten that ceiling. This is separate from the environment's episode timeout;
a longer episode timeout does not extend the worker ceiling. The default worker ceiling is below the
default 21600-second episode timeout. The episode budget covers seeding, runtime installation, agent execution,
agent close, and verification, while the worker ceiling starts after seeding. The episode deadline can therefore
fire first. Final cleanup runs after the episode deadline under its own `cleanup_timeout_seconds` budget per callback.
`session_close_timeout_seconds` budgets each cleanup/capture phase and the remaining activation join
(default: 30 seconds per phase). Both settings must be positive and finite.

See [NOOA sessions and SWE-bench Pro](../../fern/versions/latest/pages/evaluation-tutorials/nooa-swe-bench-pro.mdx)
for setup, recipes, service lifecycle and observation limits.

Sandboxed mode requires the Resources server to provide `sandbox_access`; otherwise seeding returns HTTP 422.
Its provider must support `AsyncSandbox.connect`. The Gym model server address must be reachable from inside
the task sandbox: configure `use_absolute_ip` / `default_host` for the deployment instead of a controller-only
loopback address. Runtime packaging runs in a thread so seeding does not block other agent requests.

Close raises if the worker stop is unconfirmed or the remaining activation cannot be joined within its budget;
the session stays available for a close retry. The bounded join also applies to embedded mode. An activation
interrupted by session close returns HTTP 409 to a caller whose request was not itself cancelled.


## Sandbox worker supervision

`sandbox_runner_timeout_seconds` bounds the worker execution budget; task metadata can shorten it. At the budget
the supervisor sends SIGTERM and waits a short grace before SIGKILL; the worker uses that window to stop the agent
and write the episode it has with stop reason `timeout_exceeded`. Gym grades that checkpoint as `incomplete`
(the sandbox state is what the verifier sees), but only when the supervisor's receipt vouches for the stop: a
`timeout_exceeded` result without a timed-out receipt, or a worker that left no result, is an infrastructure
failure (HTTP 504 at the budget, otherwise an error with the output-log tail), never a retryable model outage.
