from __future__ import annotations
import os, re, subprocess
from pathlib import Path

SSH_DIR = Path.home() / ".ssh"
MAIN = SSH_DIR / "config"
INC = SSH_DIR / "config.runpod_cli"
KH = SSH_DIR / "known_hosts.runpod_cli"
INCLUDE = "Include ~/.ssh/config.runpod_cli\n"

_BLOCK = re.compile(r"(?ms)^(Host[ \t]+(?P<alias>\S+)\n(?:[ \t].*\n)+)")

def ensure_include():
    SSH_DIR.mkdir(parents=True, exist_ok=True)
    MAIN.touch(exist_ok=True)
    if not MAIN.read_text(encoding="utf-8", errors="ignore").startswith(INCLUDE):
        tmp = MAIN.with_suffix(".tmp")
        tmp.write_text(INCLUDE + MAIN.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        os.replace(tmp, MAIN)
    INC.touch(exist_ok=True)
    KH.touch(exist_ok=True)
    for p in (MAIN, INC, KH):
        try: os.chmod(p, 0o600)
        except Exception: pass

def _upsert(cfg: Path, alias: str, block: str):
    s = cfg.read_text(encoding="utf-8", errors="ignore")
    out, i, replaced = [], 0, False
    for m in _BLOCK.finditer(s):
        a = m.group("alias"); start, end = m.span()
        out.append(s[i:start])
        out.append(block if a == alias else s[start:end])
        replaced |= (a == alias)
        i = end
    out.append(s[i:])
    if not replaced:
        if not s.endswith("\n"): out.append("\n")
        out.append(block)
    tmp = cfg.with_suffix(".tmp")
    tmp.write_text("".join(out), encoding="utf-8"); os.replace(tmp, cfg)

def add_known_host(ip: str, port: int):
    try:
        r = subprocess.run(["ssh-keyscan", "-p", str(port), "-T", "5", ip],
                           capture_output=True, text=True, check=False)
        if r.stdout:
            with KH.open("a", encoding="utf-8") as f: f.write(r.stdout)
    except Exception: pass  # best-effort

def render_host_block(alias: str, ip: str, port: int, user: str, identity_file: str | None = None, forward_agent: bool = True) -> str:
    lines = [
        f"Host {alias}",
        f"    HostName {ip}",
        f"    Port {port}",
        f"    User {user}",
        "    UserKnownHostsFile ~/.ssh/known_hosts ~/.ssh/known_hosts.runpod_cli",
        "    StrictHostKeyChecking yes",
    ]
    if identity_file:
        lines.insert(4, f"    IdentityFile {identity_file}")
        lines.append("    IdentitiesOnly yes")
    if forward_agent:
        lines.append("    ForwardAgent yes")
    return "\n".join(lines) + "\n"

def upsert_host(alias: str, ip: str, port: int, user: str,
                identity_file: str | None = None, mirror_to_main: bool = False, forward_agent: bool = True):
    ensure_include()
    block = render_host_block(alias, ip, port, user, identity_file, forward_agent)
    _upsert(INC, alias, block)
    add_known_host(ip, port)
    if mirror_to_main:
        _upsert(MAIN, alias, block)

def remove_host(alias: str, also_from_main: bool = False):
    def _rm(cfg: Path):
        s = cfg.read_text(encoding="utf-8", errors="ignore")
        out, i = [], 0
        for m in _BLOCK.finditer(s):
            start, end = m.span()
            if m.group("alias") == alias:
                out.append(s[i:start]); i = end
        out.append(s[i:])
        if out and "".join(out) != s:
            tmp = cfg.with_suffix(".tmp")
            tmp.write_text("".join(out), encoding="utf-8"); os.replace(tmp, cfg)
    if INC.exists(): _rm(INC)
    if also_from_main and MAIN.exists(): _rm(MAIN)

# --- Windows / WSL helpers (for Cursor on Windows) ---
def _is_wsl() -> bool:
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            s = f.read().lower()
        return ("microsoft" in s) or ("wsl" in s)
    except Exception:
        return False

def _windows_home_from_cmd() -> Path | None:
    try:
        r = subprocess.run(["cmd.exe", "/c", "echo", "%USERPROFILE%"], capture_output=True, text=True, check=False)
        raw = r.stdout.strip().replace("\r", "")
        if not raw:
            return None
        drive, rest = raw.split(":", 1)
        return Path(f"/mnt/{drive.lower()}{rest.replace('\\', '/')}")  # /mnt/c/Users/<You>
    except Exception:
        return None

def _win_ssh_paths() -> tuple[Path, Path, Path] | None:
    if not _is_wsl():
        return None
    home = _windows_home_from_cmd()
    if home is None:
        return None
    ssh_dir = home / ".ssh"
    cfg = ssh_dir / "config"
    kh_extra = ssh_dir / "known_hosts.runpod_cli"
    return (ssh_dir, cfg, kh_extra)

def ensure_windows_has_private_key(identity_name: str = "id_ed25519") -> str | None:
    """
    Copy WSL ~/.ssh/<identity_name>[.pub] into Windows' ~/.ssh.
    Returns the Windows-side IdentityFile path (OpenSSH ~-style), or None on failure.
    """
    paths = _win_ssh_paths()
    if paths is None:
        return None
    ssh_dir, _, _ = paths

    src_priv = (SSH_DIR / identity_name)
    src_pub  = (SSH_DIR / f"{identity_name}.pub")
    if not src_priv.exists() or not src_pub.exists():
        # keypair absent in WSL; don't try to create one silently here
        return None

    ssh_dir.mkdir(parents=True, exist_ok=True)
    dst_priv = ssh_dir / identity_name
    dst_pub  = ssh_dir / f"{identity_name}.pub"

    try:
        # best-effort copy; don't overwrite without prompt in future if you prefer
        dst_priv.write_bytes(src_priv.read_bytes())
        dst_pub.write_bytes(src_pub.read_bytes())
        try:
            os.chmod(dst_priv, 0o600)
        except Exception:
            pass
        # Windows OpenSSH interprets "~/.ssh/..." in its own HOME
        return "~/.ssh/" + identity_name
    except Exception:
        return None

def upsert_host_windows(alias: str, ip: str, port: int, user: str, identity_file: str | None = None, forward_agent: bool = True) -> None:
    paths = _win_ssh_paths()
    if paths is None:
        return
    ssh_dir, cfg, _ = paths
    ssh_dir.mkdir(parents=True, exist_ok=True)
    block = render_host_block(alias=alias, ip=ip, port=port, user=user, identity_file=identity_file, forward_agent=forward_agent)
    _upsert(cfg, alias, block)
    try:
        os.chmod(cfg, 0o600)
    except Exception:
        pass

def copy_known_hosts_to_windows() -> None:
    paths = _win_ssh_paths()
    if paths is None:
        return
    _, _, win_kh = paths
    try:
        if KH.exists():
            # append if not already present
            add = KH.read_text(encoding="utf-8", errors="ignore")
            existing = win_kh.read_text(encoding="utf-8", errors="ignore") if win_kh.exists() else ""
            if add and add not in existing:
                with win_kh.open("a", encoding="utf-8") as f:
                    f.write(add)
        try:
            os.chmod(win_kh, 0o600)
        except Exception:
            pass
    except Exception:
        pass

def remove_host_windows(alias: str) -> None:
    paths = _win_ssh_paths()
    if paths is None:
        return
    _, cfg, _ = paths
    if not cfg.exists():
        return
    s = cfg.read_text(encoding="utf-8", errors="ignore")
    out, i = [], 0
    for m in _BLOCK.finditer(s):
        start, end = m.span()
        if m.group("alias") == alias:
            out.append(s[i:start]); i = end
    out.append(s[i:])
    new = "".join(out)
    if new != s:
        tmp = cfg.with_suffix(".tmp")
        tmp.write_text(new, encoding="utf-8")
        os.replace(tmp, cfg)

