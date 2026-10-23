# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and stage a relocatable NOOA runtime for the task sandbox.

Build (once per architecture, outside any task):
    python -m responses_api_agents.nooa_thin_agent.runtime build --output nooa-runtime-x86_64.tar.gz \
        --arch x86_64 <nooa source dirs or requirement specs...>

The tarball is a python-build-standalone interpreter with NOOA installed into
its own site-packages. Nothing runs ``pip`` inside the sandbox: staging is an
upload, ``tar -x``, and one import check, cached per tarball digest.
"""

from __future__ import annotations

import argparse
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
PYTHON_RELEASE = "20260310"
PYTHON_VERSION = "3.13.12"
PYTHON_SHA256 = {
    "x86_64": "904adc9bc4371c01b20f7e75a19b10f07fb577889be02bf21f7f449229b97611",  # pragma: allowlist secret
    "aarch64": "0ebc0049121318b5de80b887d22abaed55dd302014f73cd6811c9981a83d960e",  # pragma: allowlist secret
}
REMOTE_ROOT = "/tmp/nemo-gym-nooa-runtime"
# cryptography is excluded by Gym's root resolver policy; the import proves the build used uv.toml.
HEALTH_CHECK = "import cryptography, mcp, nooa, nooa_bench.gym_worker"
# Replaces uv's config discovery during the build: inside the Gym checkout the root
# [tool.uv] exclude-dependencies would silently drop cryptography from the tarball.
UV_CONFIG = Path(__file__).with_name("uv.toml")


def python_url(arch: str) -> str:
    return (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{PYTHON_RELEASE}/cpython-{PYTHON_VERSION}%2B{PYTHON_RELEASE}-{arch}-unknown-linux-gnu-install_only_stripped.tar.gz"
    )


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_runtime(output: Path, *, arch: str, requirements: list[str], cache: Path | None = None) -> Path:
    """Produce ``output``: ``python/`` with NOOA and its dependencies installed."""
    if arch not in PYTHON_SHA256:
        raise ValueError(f"unsupported architecture {arch!r}; expected one of {sorted(PYTHON_SHA256)}")
    cache = cache or Path(tempfile.gettempdir()) / "nemo-gym-nooa-runtime-assets"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / python_url(arch).rsplit("/", 1)[1]
    if not archive.is_file() or file_sha256(archive) != PYTHON_SHA256[arch]:
        with urllib.request.urlopen(python_url(arch), timeout=120) as response:  # noqa: S310 (pinned https URL)
            archive.write_bytes(response.read())
        if file_sha256(archive) != PYTHON_SHA256[arch]:
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
        native = platform.machine() == arch
        config = ["--config-file", str(UV_CONFIG)]
        if native:
            install = [uv, "pip", "install", *config, "--quiet", "--python", str(python), "--break-system-packages", *specs]
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
                f"{arch}-unknown-linux-gnu",
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


async def stage_runtime(sandbox: AsyncSandbox, tarball: str | Path) -> str:
    """Make the runtime available in ``sandbox`` and return its interpreter path.

    Keyed by the tarball digest: a sandbox that already holds a healthy copy is reused.
    """
    tarball = Path(tarball)
    digest = file_sha256(tarball)
    target = f"{REMOTE_ROOT}/{digest[:16]}"
    python = f"{target}/python/bin/python3"
    check = f"test -x {shlex.quote(python)} && {shlex.quote(python)} -I -c {shlex.quote(HEALTH_CHECK)}"
    healthy = await sandbox.exec(check, cwd="/", timeout_s=60)
    if healthy.return_code == 0 and not healthy.error_type:
        return python
    staged = f"{target}.tar.gz"
    await sandbox.exec(f"mkdir -p {shlex.quote(REMOTE_ROOT)}", cwd="/", timeout_s=30)
    await sandbox.upload(tarball, staged)
    install = (
        f"set -e; rm -rf {shlex.quote(target + '.tmp')}; mkdir -p {shlex.quote(target + '.tmp')}; "
        f"tar -xzf {shlex.quote(staged)} -C {shlex.quote(target + '.tmp')}; "
        f"rm -f {shlex.quote(staged)}; mv -T {shlex.quote(target + '.tmp')} {shlex.quote(target)}; {check}"
    )
    result = await sandbox.exec(install, cwd="/", timeout_s=600)
    if result.return_code != 0 or result.error_type:
        raise RuntimeError(f"NOOA runtime staging failed (exit {result.return_code}): {(result.stderr or '')[-2000:]}")
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--arch", choices=sorted(PYTHON_SHA256), required=True)
    build.add_argument("requirements", nargs="+", help="pip requirement specs or local source directories")
    args = parser.parse_args()
    path = build_runtime(args.output, arch=args.arch, requirements=args.requirements)
    print(f"{path} sha256={file_sha256(path)}")


if __name__ == "__main__":
    main()
