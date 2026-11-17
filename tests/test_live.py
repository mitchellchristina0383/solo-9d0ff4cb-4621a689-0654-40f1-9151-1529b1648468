"""Live multi-process tests: two uvicorn processes share one data dir.

Covers: concurrent chunk uploads (incl. duplicate and conflicting content),
concurrent finalize from both processes, kill/restart mid-upload, and that
published packages remain downloadable after restarts. Also: two processes
comparing the same task pair concurrently leave exactly one record, a
concurrently failing comparison leaves none, and records survive restarts.

Run:  .venv/bin/python -m pytest tests/test_live.py -v
"""
import hashlib
import io
import os
import random
import socket
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    def __init__(self, data_dir: Path, log_dir: Path):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ, DATA_DIR=str(data_dir), PYTHONPATH=str(ROOT))
        self.log = open(log_dir / f"server-{self.port}.log", "wb")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", str(self.port)],
            env=env, cwd=str(ROOT), stdout=self.log, stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server on {self.port} exited early")
            try:
                if httpx.get(self.base + "/api/health", timeout=1).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"server on {self.port} not ready")

    def kill(self):
        self.proc.kill()
        self.proc.wait()
        self.log.close()

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.log.close()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def create_task(client: httpx.Client, base: str, data: bytes) -> str:
    r = client.post(f"{base}/api/tasks", json={
        "filename": "diag.tar.gz", "size": len(data), "sha256": sha256(data)})
    assert r.status_code == 201, r.text
    return r.json()["task_id"]


def put_chunk(client: httpx.Client, base: str, tid: str, i: int, payload: bytes):
    return client.put(f"{base}/api/tasks/{tid}/chunks/{i}", content=payload)


def test_two_processes_concurrent_upload_and_finalize(tmp_path):
    data_dir = tmp_path / "data"
    s1, s2 = Server(data_dir, tmp_path), Server(data_dir, tmp_path)
    try:
        s1.wait_ready()
        s2.wait_ready()
        data = random.Random(7).randbytes(5 * CHUNK + 999)  # 6 chunks
        chunks = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)]

        with httpx.Client(timeout=30) as client:
            tid = create_task(client, s1.base, data)

            # 两个进程并发上传：正常分块 + 同内容重复分块
            jobs = []
            for i, c in enumerate(chunks):
                jobs.append((s1.base if i % 2 == 0 else s2.base, i, c))
            # 同内容重传（打到另一个进程）
            for i in range(3):
                jobs.append((s2.base if i % 2 == 0 else s1.base, i, chunks[i]))

            def send(job):
                base, i, payload = job
                return put_chunk(client, base, tid, i, payload)

            with ThreadPoolExecutor(max_workers=9) as ex:
                results = list(ex.map(send, jobs))
            assert all(r.status_code == 200 for r in results), \
                [r.text for r in results]

            st = client.get(f"{s2.base}/api/tasks/{tid}").json()
            assert st["received_count"] == 6 and st["missing_chunks"] == []

            # 异内容重传：拒绝且保留原块
            evil = bytearray(chunks[0])
            evil[0] ^= 0xFF
            r = put_chunk(client, s2.base, tid, 0, bytes(evil))
            assert r.status_code == 409

            # 两个进程并发提交同一任务
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = [ex.submit(client.post, f"{b}/api/tasks/{tid}/finalize")
                        for b in (s1.base, s2.base)]
                finals = [f.result() for f in futs]
            assert all(r.status_code == 200 for r in finals), \
                [r.text for r in finals]
            assert all(r.json()["state"] == "published" for r in finals)

            # 两个进程下载到的成品一致且正确；磁盘上只有一份成品
            for base in (s1.base, s2.base):
                dl = client.get(f"{base}/api/tasks/{tid}/download")
                assert dl.status_code == 200 and dl.content == data
            published = list((data_dir / "published").iterdir())
            assert len(published) == 1
            assert published[0].read_bytes() == data
    finally:
        s1.stop()
        s2.stop()


def test_kill_restart_mid_upload_and_published_survives(tmp_path):
    data_dir = tmp_path / "data"
    data = random.Random(11).randbytes(3 * CHUNK + 1)  # 4 chunks
    chunks = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)]

    s1 = Server(data_dir, tmp_path)
    s1.wait_ready()
    with httpx.Client(timeout=30) as client:
        tid = create_task(client, s1.base, data)
        assert put_chunk(client, s1.base, tid, 0, chunks[0]).status_code == 200
        assert put_chunk(client, s1.base, tid, 1, chunks[1]).status_code == 200
    s1.kill()  # 模拟进程崩溃（kill -9）

    s2 = Server(data_dir, tmp_path)
    try:
        s2.wait_ready()
        with httpx.Client(timeout=30) as client:
            st = client.get(f"{s2.base}/api/tasks/{tid}").json()
            assert st["received_chunks"] == [0, 1]  # 已收分块保留
            assert put_chunk(client, s2.base, tid, 2, chunks[2]).status_code == 200
            assert put_chunk(client, s2.base, tid, 3, chunks[3]).status_code == 200
            r = client.post(f"{s2.base}/api/tasks/{tid}/finalize")
            assert r.status_code == 200 and r.json()["state"] == "published"
    finally:
        s2.kill()  # 发布后再崩溃

    s3 = Server(data_dir, tmp_path)
    try:
        s3.wait_ready()
        with httpx.Client(timeout=30) as client:
            dl = client.get(f"{s3.base}/api/tasks/{tid}/download")
            assert dl.status_code == 200 and dl.content == data  # 已发布仍可下载
    finally:
        s3.stop()


def test_concurrent_duplicate_same_chunk_across_processes(tmp_path):
    """Same chunk, same content, raced from two processes: all 200, one record."""
    data_dir = tmp_path / "data"
    s1, s2 = Server(data_dir, tmp_path), Server(data_dir, tmp_path)
    try:
        s1.wait_ready()
        s2.wait_ready()
        data = random.Random(3).randbytes(CHUNK)
        with httpx.Client(timeout=30) as client:
            tid = create_task(client, s1.base, data)
            with ThreadPoolExecutor(max_workers=2) as ex:
                r1 = ex.submit(put_chunk, client, s1.base, tid, 0, data)
                r2 = ex.submit(put_chunk, client, s2.base, tid, 0, data)
                results = [r1.result(), r2.result()]
            assert all(r.status_code == 200 for r in results)
            st = client.get(f"{s1.base}/api/tasks/{tid}").json()
            assert st["received_chunks"] == [0]
            assert client.post(f"{s2.base}/api/tasks/{tid}/finalize").json()[
                "state"] == "published"
            assert client.get(f"{s1.base}/api/tasks/{tid}/download").content == data
    finally:
        s1.stop()
        s2.stop()


def make_zip(entries) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


def publish(client: httpx.Client, base: str, data: bytes, filename: str) -> str:
    r = client.post(f"{base}/api/tasks", json={
        "filename": filename, "size": len(data), "sha256": sha256(data)})
    assert r.status_code == 201, r.text
    tid = r.json()["task_id"]
    for i in range(0, len(data), CHUNK):
        resp = client.put(f"{base}/api/tasks/{tid}/chunks/{i // CHUNK}",
                          content=data[i:i + CHUNK])
        assert resp.status_code == 200, resp.text
    r = client.post(f"{base}/api/tasks/{tid}/finalize")
    assert r.status_code == 200 and r.json()["state"] == "published", r.text
    return tid


def test_two_processes_concurrent_compare_single_record(tmp_path):
    """Two processes comparing the same pair at once -> exactly one record;
    a concurrently failing comparison -> no record at all."""
    data_dir = tmp_path / "data"
    s1, s2 = Server(data_dir, tmp_path), Server(data_dir, tmp_path)
    try:
        s1.wait_ready()
        s2.wait_ready()
        with httpx.Client(timeout=60) as client:
            zb = make_zip([("a.log", b"one"), ("b.log", b"two")])
            zr = make_zip([("a.log", b"one"), ("b.log", b"TWO"), ("c.log", b"3")])
            junk = random.Random(21).randbytes(4096)
            # 两个进程各发布一部分，互相可见
            b = publish(client, s1.base, zb, "baseline.zip")
            r = publish(client, s2.base, zr, "review.zip")
            j = publish(client, s1.base, junk, "junk.bin")

            # 同一对任务，两个进程同时核对
            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = [ex.submit(client.post, f"{base}/api/comparisons",
                                     json={"baseline_id": b, "review_id": r})
                           for base in (s1.base, s2.base, s1.base, s2.base)]
                results = [f.result() for f in futures]
            assert all(x.status_code in (200, 201) for x in results), \
                [x.text for x in results]
            ids = {x.json()["comparison_id"] for x in results}
            assert len(ids) == 1  # 只留下同一份结果
            body = results[0].json()
            assert body["added"] == ["c.log"] and body["changed"] == ["b.log"]
            for base in (s1.base, s2.base):
                lst = client.get(f"{base}/api/comparisons").json()["comparisons"]
                assert len(lst) == 1

            # 同一对失败任务，两个进程同时核对：全部失败且不留任何记录
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = [ex.submit(client.post, f"{base}/api/comparisons",
                                     json={"baseline_id": b, "review_id": j})
                           for base in (s1.base, s2.base)]
                fails = [f.result() for f in futures]
            assert all(x.status_code == 422 for x in fails), \
                [x.text for x in fails]
            assert all(x.json()["error"]["code"] == "bad_zip" for x in fails)
            for base in (s1.base, s2.base):
                got = client.get(f"{base}/api/comparisons/pair",
                                 params={"baseline_id": b, "review_id": j})
                assert got.status_code == 404
            lst = client.get(f"{s1.base}/api/comparisons").json()["comparisons"]
            assert len(lst) == 1  # 仍只有成功那一份
    finally:
        s1.stop()
        s2.stop()

    # 重启后核对记录仍在
    s3 = Server(data_dir, tmp_path)
    try:
        s3.wait_ready()
        with httpx.Client(timeout=30) as client:
            lst = client.get(f"{s3.base}/api/comparisons").json()["comparisons"]
            assert len(lst) == 1
            assert lst[0]["changed"] == ["b.log"]
    finally:
        s3.stop()
