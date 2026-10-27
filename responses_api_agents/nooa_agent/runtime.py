# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and stage a relocatable NOOA runtime for the task sandbox.

Build (once per target, outside any task):
    python -m responses_api_agents.nooa_agent.runtime build --output nooa-runtime-x86_64.tar.gz \
        --arch x86_64 <nooa source dirs or requirement specs...>
    python -m responses_api_agents.nooa_agent.runtime build --output nooa-runtime-x86_64-musl.tar.gz \
        --target x86_64-unknown-linux-musl <same requirements...>

The tarball is a python-build-standalone interpreter with NOOA installed into
its own site-packages. Nothing runs ``pip`` inside the sandbox: staging probes
the image's libc (glibc or musl), picks the matching tarball, then does an
upload, ``tar -x``, and one import check, cached per tarball digest.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import platform
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox


# Same artifacts the mini-SWE adapter pins (uv 0.10.12's manifest). NOOA requires >=3.12.
# Keyed by target triple: task images are glibc (Debian/Ubuntu) or musl (Alpine), and a
# glibc interpreter cannot exec on musl (the shell reports exit 127).
PYTHON_RELEASE = "20260310"
PYTHON_VERSION = "3.13.12"
PYTHON_SHA256 = {
    "x86_64-unknown-linux-gnu": "904adc9bc4371c01b20f7e75a19b10f07fb577889be02bf21f7f449229b97611",  # pragma: allowlist secret
    "aarch64-unknown-linux-gnu": "0ebc0049121318b5de80b887d22abaed55dd302014f73cd6811c9981a83d960e",  # pragma: allowlist secret
    "x86_64-unknown-linux-musl": "3c9db1ed094d6d08e474600b7a4eab7cab655b60f7f74c3a41f1d995eb53611a",  # pragma: allowlist secret
}
REMOTE_ROOT = "/tmp/nemo-gym-nooa-runtime"
# Task images may ship a PATH without the system bin dirs (every command is then "not found",
# exit 127); staging and the health check run with the standard directories in front.
SANE_PATH = "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH};"
# One shell line, as nemo_gym.agent_utils.uv_bootstrap probes it: machine, then C library.
LIBC_PROBE = "uname -m && if ls /lib/ld-musl-*.so.1 >/dev/null 2>&1; then echo musl; else echo gnu; fi"
# cryptography is excluded by Gym's root resolver policy; the import proves the build used uv.toml.
HEALTH_CHECK = "import cryptography, mcp, nooa, nooa_bench.gym_worker"
# Replaces uv's config discovery during the build: inside the Gym checkout the root
# [tool.uv] exclude-dependencies would silently drop cryptography from the tarball.
UV_CONFIG = Path(__file__).with_name("uv.toml")


def python_url(target: str) -> str:
    return (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{PYTHON_RELEASE}/cpython-{PYTHON_VERSION}%2B{PYTHON_RELEASE}-{target}-install_only_stripped.tar.gz"
    )


def host_target() -> str:
    # platform.libc_ver() reports '' on musl hosts, so look for the loader as the sandbox probe does.
    libc = "musl" if glob.glob("/lib/ld-musl-*.so.1") else "gnu"
    return f"{platform.machine()}-unknown-linux-{libc}"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_runtime(output: Path, *, target: str, requirements: list[str], cache: Path | None = None) -> Path:
    """Produce ``output``: ``python/`` with NOOA and its dependencies installed, for ``target``."""
    if target not in PYTHON_SHA256:
        raise ValueError(f"unsupported target {target!r}; expected one of {sorted(PYTHON_SHA256)}")
    cache = cache or Path(tempfile.gettempdir()) / "nemo-gym-nooa-runtime-assets"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / python_url(target).rsplit("/", 1)[1]
    if not archive.is_file() or file_sha256(archive) != PYTHON_SHA256[target]:
        with urllib.request.urlopen(python_url(target), timeout=120) as response:  # noqa: S310 (pinned https URL)
            archive.write_bytes(response.read())
        if file_sha256(archive) != PYTHON_SHA256[target]:
            raise RuntimeError("python-build-standalone checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="nooa-runtime-build-") as directory:
        build = Path(directory)
        with tarfile.open(archive) as tar:
            tar.extractall(build, filter="data")  # yields build/python
        python = build / "python" / "bin" / "python3"
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("building the runtime requires uv on PATH")
        # Local source directories become wheels first: a direct path install can resolve to
        # an editable install that points back at the build host.
        wheels = build / "wheels"
        wheels.mkdir()
        specs: list[str] = []
        for requirement in requirements:
            if Path(requirement).is_dir():
                subprocess.run([uv, "build", "--quiet", "--wheel", "--out-dir", str(wheels), requirement], check=True)
            else:
                specs.append(requirement)
        specs.extend(str(wheel) for wheel in sorted(wheels.glob("*.whl")))
        # Install into the interpreter's own site-packages: no venv, nothing to relocate but the tree.
        native = host_target() == target
        config = ["--config-file", str(UV_CONFIG)]
        if native:
            install = [
                uv,
                "pip",
                "install",
                *config,
                "--quiet",
                "--python",
                str(python),
                "--break-system-packages",
                *specs,
            ]
        else:
            # Cross-build (for example x86_64 pods from an aarch64 controller): resolve wheels for the
            # target platform without running its interpreter. Everything must ship as a wheel.
            site = next((build / "python" / "lib").glob("python3.*")) / "site-packages"
            install = [
                uv,
                "pip",
                "install",
                *config,
                "--quiet",
                "--target",
                str(site),
                "--only-binary",
                ":all:",
                "--python-version",
                PYTHON_VERSION,
                "--python-platform",
                target,
                *specs,
            ]
        subprocess.run(install, check=True)
        for marker in (build / "python").rglob("_editable_impl_*.pth"):
            raise RuntimeError(f"editable install leaked into the runtime: {marker.name}")
        if native:
            subprocess.run([str(python), "-I", "-c", HEALTH_CHECK], check=True)
        else:
            # Can't execute the target interpreter here; the sandbox-side import check in
            # stage_runtime() is the health check for cross-built tarballs.
            for package in ("nooa", "nooa_bench", "litellm", "httpx", "mcp", "cryptography", "cffi"):
                if not (site / package).is_dir():
                    raise RuntimeError(f"cross-build is missing package directory {package!r}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as tar:
            tar.add(build / "python", arcname="python")
    return output


async def probe_libc(sandbox: AsyncSandbox) -> tuple[str, str]:
    """``(machine, libc)`` of the sandbox image, libc being ``gnu`` or ``musl``."""
    result = await sandbox.exec(f"{SANE_PATH} {LIBC_PROBE}", cwd="/", timeout_s=30)
    lines = (result.stdout or "").split()
    if result.return_code != 0 or len(lines) < 2:
        raise RuntimeError(f"could not probe the sandbox platform (exit {result.return_code}): {result.stderr or ''}")
    return lines[0], lines[1]


async def stage_runtime(sandbox: AsyncSandbox, tarball: str | Path, *, musl_tarball: str | Path | None = None) -> str:
    """Make the runtime available in ``sandbox`` and return its interpreter path.

    ``tarball`` is the glibc build; ``musl_tarball`` serves Alpine-style images, which cannot
    exec a glibc interpreter. Keyed by the tarball digest: a sandbox that already holds a
    healthy copy is reused.
    """
    machine, libc = await probe_libc(sandbox)
    if libc == "musl":
        if musl_tarball is None:
            raise RuntimeError(
                f"sandbox image uses musl libc ({machine}) and no musl NOOA runtime is configured; "
                "build one with `runtime build --target x86_64-unknown-linux-musl` and set runtime_tarball_musl"
            )
        tarball = Path(musl_tarball)
    else:
        tarball = Path(tarball)
    digest = file_sha256(tarball)
    target = f"{REMOTE_ROOT}/{digest[:16]}"
    python = f"{target}/python/bin/python3"
    check = f"test -x {shlex.quote(python)} && {shlex.quote(python)} -I -c {shlex.quote(HEALTH_CHECK)}"
    healthy = await sandbox.exec(f"{SANE_PATH} {check}", cwd="/", timeout_s=60)
    if healthy.return_code == 0 and not healthy.error_type:
        return python
    staged = f"{target}.tar.gz"
    await sandbox.exec(f"{SANE_PATH} mkdir -p {shlex.quote(REMOTE_ROOT)}", cwd="/", timeout_s=30)
    await sandbox.upload(tarball, staged)
    tmp = shlex.quote(target + ".tmp")
    # Each step is named on stderr, so a bare exit code (127: command or loader not found) says
    # which command failed; a failed health check also dumps what the image offers.
    steps = [
        ("clean", f"rm -rf {tmp}"),
        ("mkdir", f"mkdir -p {tmp}"),
        ("extract", f"tar -xzf {shlex.quote(staged)} -C {tmp}"),
        ("discard-archive", f"rm -f {shlex.quote(staged)}"),
        # No `mv -T`: busybox (Alpine images) lacks it. The target is absent or a failed earlier
        # staging; removing it first makes a plain mv a rename either way.
        ("rename", f"rm -rf {shlex.quote(target)} && mv {tmp} {shlex.quote(target)}"),
        ("health-check", check),
    ]
    diagnostics = (
        "echo '--- diagnostics' >&2; "
        f"(uname -m; {LIBC_PROBE}) 2>&1 | tr '\\n' ' ' >&2; echo >&2; "
        'echo "PATH=$PATH" >&2; command -v sh bash tar gzip mv mkdir rm >&2 2>&1; '
        f"ls -la {shlex.quote(python)} {shlex.quote(target + '/python/bin')} >&2 2>&1 | head -5; "
        f"(ldd {shlex.quote(python)} || true) 2>&1 | head -5 >&2; "
        "(df -k /tmp; mount | grep -w /tmp) 2>&1 | tail -3 >&2"
    )
    script = (
        SANE_PATH
        + " "
        + " ".join(
            f"echo '+ {name}' >&2; {command} || {{ rc=$?; echo \"step {name} failed (exit $rc)\" >&2; {diagnostics}; exit $rc; }};"
            for name, command in steps
        )
    )
    result = await sandbox.exec(script, cwd="/", timeout_s=600)
    if result.return_code != 0 or result.error_type:
        detail = ((result.stderr or "") + "\n" + (result.stdout or ""))[-4000:]
        raise RuntimeError(f"NOOA runtime staging failed (exit {result.return_code}): {detail}")
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    group = build.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", choices=sorted(PYTHON_SHA256), help="python-build-standalone target triple")
    group.add_argument("--arch", choices=["x86_64", "aarch64"], help="shorthand for the glibc target of this machine")
    build.add_argument("requirements", nargs="+", help="pip requirement specs or local source directories")
    args = parser.parse_args()
    target = args.target or f"{args.arch}-unknown-linux-gnu"
    path = build_runtime(args.output, target=target, requirements=args.requirements)
    print(f"{path} sha256={file_sha256(path)}")


if __name__ == "__main__":
    main()
