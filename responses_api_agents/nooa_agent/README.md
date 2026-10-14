# NOOA Agent

The NOOA agent adapter provides typed configuration for constructing NOOA agents and invoking them through
asynchronous Python adapters that receive validated NeMo Gym Responses requests.

## Embedded CodeAct scope

Embedded CodeAct persists for the duration of one invocation. `execute_python` is not computation-only: it can
read and modify files, start processes, use the network, and change other node state. Embedded mode provides no
sandbox protection. Only injected resource methods are Gym/verifier-observable and authoritative; direct CodeAct
side effects belong to NOOA and are not reliably visible to verification or replay. The trajectory distinguishes
the outer `execute_python` call from nested resource calls.

## Sandboxed execution

`nooa.execution_mode: sandboxed` runs the complete NOOA agent in the task sandbox supplied by Resources.

Set `retain_task_services_until_close: true` (sandboxed mode only) to keep services the agent started
(for example `nohup python3 -m http.server &`) alive through verification. The supervisor then writes
`completion.json` when the worker exits and, only after a successful in-deadline exit, keeps its Linux
subreaper alive until session close touches `runner.stop`. `nooa_single_agent_turn` already finishes the
session before `/verify` and closes it afterwards. Worker failures, missing results, the agent deadline and
cancellation still stop and reap every descendant immediately. The default (`false`) keeps the old
behaviour: everything is reaped when the worker exits.
NOOA stages its own runtime and closes its worker before disconnecting; Resources owns sandbox creation and teardown.
Native sessions use the NOOA-specific `nooa_single_agent_turn` environment.

See [NOOA sessions and SWE-bench Pro](../../fern/versions/latest/pages/evaluation-tutorials/nooa-swe-bench-pro.mdx)
for setup, recipes, service lifecycle and observation limits.
