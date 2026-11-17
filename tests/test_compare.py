"""Functional tests for ZIP package comparison (TestClient).

Covers: content diff (added/deleted/changed), metadata-only differences,
artifact integrity re-verification, all archive-defect failures leaving no
record, pair-pinning/idempotency, restart persistence, and concurrent
same-pair requests (success and failure).
"""
import hashlib
import io
import random
import struct
import sys
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.compare import MAX_ENTRIES, MAX_UNCOMPRESSED  # noqa: E402
from app.main import CHUNK_SIZE, create_app  # noqa: E402


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "data"))


def publish(client: TestClient, data: bytes, filename="pkg.zip") -> str:
    r = client.post(
        "/api/tasks",
        json={"filename": filename, "size": len(data), "sha256": sha256(data)},
    )
    assert r.status_code == 201, r.text
    tid = r.json()["task_id"]
    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i : i + CHUNK_SIZE]
        resp = client.put(f"/api/tasks/{tid}/chunks/{i // CHUNK_SIZE}", content=chunk)
        assert resp.status_code == 200, resp.text
    r = client.post(f"/api/tasks/{tid}/finalize")
    assert r.status_code == 200 and r.json()["state"] == "published", r.text
    return tid


def make_zip(entries, compress=zipfile.ZIP_DEFLATED, dt=(2024, 1, 1, 0, 0, 0)) -> bytes:
    """entries: list of (name, bytes|None); None makes a directory entry."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=dt)
            info.compress_type = compress
            if data is None:
                info.external_attr = 0o40755 << 16
                zf.writestr(info, b"")
            else:
                zf.writestr(info, data)
    return buf.getvalue()


def compare(client: TestClient, baseline_id: str, review_id: str):
    return client.post(
        "/api/comparisons",
        json={"baseline_id": baseline_id, "review_id": review_id},
    )


def pair_of(client: TestClient, baseline_id: str, review_id: str):
    return client.get(
        f"/api/comparisons/pair?baseline_id={baseline_id}&review_id={review_id}"
    )


BASE_FILES = [
    ("logs/app.log", b"v1\n" * 100),
    ("logs/sys.log", b"sys\n" * 50),
    ("meta/info.txt", b"incident-42"),
    ("readme.txt", b"old readme"),
]
REVIEW_FILES = [
    ("logs/app.log", b"v1\n" * 100),            # unchanged
    ("logs/sys.log", b"sys CHANGED\n" * 50),    # changed content
    ("meta/info.txt", b"incident-42"),          # unchanged
    ("trace/new.log", b"new trace"),            # added
    # readme.txt deleted
]


@pytest.fixture
def pair(tmp_path):
    """A client plus one published baseline and one published review zip."""
    client = make_client(tmp_path)
    b = publish(client, make_zip(BASE_FILES), "baseline.zip")
    r = publish(client, make_zip(REVIEW_FILES), "review.zip")
    return client, b, r


def test_compare_added_deleted_changed(pair):
    client, b, r = pair
    resp = compare(client, b, r)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["state"] == "done"
    assert body["added"] == ["trace/new.log"]
    assert body["deleted"] == ["readme.txt"]
    assert body["changed"] == ["logs/sys.log"]
    assert body["unchanged_count"] == 2
    assert body["counts"] == {"added": 1, "deleted": 1, "changed": 1}
    assert body["baseline"]["task_id"] == b
    assert body["review"]["task_id"] == r

    # 记录按任务对固定：pair 查询、列表、按 id 查询均返回同一份
    got = pair_of(client, b, r)
    assert got.status_code == 200
    assert got.json()["comparison_id"] == body["comparison_id"]
    lst = client.get("/api/comparisons").json()["comparisons"]
    assert [c["comparison_id"] for c in lst] == [body["comparison_id"]]
    by_id = client.get(f"/api/comparisons/{body['comparison_id']}")
    assert by_id.status_code == 200
    assert by_id.json()["added"] == ["trace/new.log"]


def test_compare_is_idempotent_single_record(pair):
    client, b, r = pair
    first = compare(client, b, r)
    assert first.status_code == 201
    again = compare(client, b, r)
    assert again.status_code == 200  # 命中已固定记录
    assert again.json()["comparison_id"] == first.json()["comparison_id"]
    lst = client.get("/api/comparisons").json()["comparisons"]
    assert len(lst) == 1


def test_compression_and_timestamp_only_differences_are_ignored(tmp_path):
    client = make_client(tmp_path)
    files = [("a.txt", b"same content"), ("dir/b.bin", bytes(range(256)) * 10)]
    z1 = make_zip(files, compress=zipfile.ZIP_STORED, dt=(2020, 5, 4, 3, 2, 1))
    z2 = make_zip(files, compress=zipfile.ZIP_DEFLATED, dt=(2026, 11, 17, 8, 9, 10))
    assert z1 != z2  # 字节层面确实不同
    b = publish(client, z1, "stored.zip")
    r = publish(client, z2, "deflated.zip")

    resp = compare(client, b, r)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["added"] == body["deleted"] == body["changed"] == []
    assert body["unchanged_count"] == 2


def test_unpublished_tasks_cannot_participate(tmp_path):
    client = make_client(tmp_path)
    published = publish(client, make_zip([("a", b"1")]))
    # 仍在 uploading 的任务
    r = client.post("/api/tasks", json={
        "filename": "pending.zip", "size": 10, "sha256": sha256(b"0123456789")})
    uploading = r.json()["task_id"]
    # failed 任务（摘要不符被拒绝发布）
    bad = client.post("/api/tasks", json={
        "filename": "bad.zip", "size": 3, "sha256": sha256(b"xxx")}).json()["task_id"]
    client.put(f"/api/tasks/{bad}/chunks/0", content=b"abc")
    assert client.post(f"/api/tasks/{bad}/finalize").status_code == 409

    for other in (uploading, bad):
        assert compare(client, published, other).status_code == 409
        assert compare(client, other, published).status_code == 409
    assert compare(client, published, "0" * 32).status_code == 404
    assert client.get("/api/comparisons").json()["comparisons"] == []


def test_integrity_mismatch_fails_and_leaves_no_record(tmp_path):
    client = make_client(tmp_path)
    b = publish(client, make_zip([("a.txt", b"alpha")]), "a.zip")
    r = publish(client, make_zip([("a.txt", b"beta")]), "b.zip")

    # 磁盘上的成品被篡改（大小不变，内容变）→ 与登记摘要不符
    pub = tmp_path / "data" / "published" / r
    raw = bytearray(pub.read_bytes())
    raw[-8] ^= 0xFF
    pub.write_bytes(bytes(raw))

    resp = compare(client, b, r)
    assert resp.status_code == 422, resp.text
    err = resp.json()["error"]
    assert err["code"] == "integrity_mismatch" and err["package"] == "review"
    assert pair_of(client, b, r).status_code == 404  # 不留记录

    # 修好后篡改大小 → 与登记大小不符
    raw2 = bytearray(pub.read_bytes())
    raw2[-8] ^= 0xFF
    pub.write_bytes(bytes(raw2) + b"x")
    resp = compare(client, b, r)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "integrity_mismatch"
    assert pair_of(client, b, r).status_code == 404


def test_artifact_missing_fails(tmp_path):
    client = make_client(tmp_path)
    b = publish(client, make_zip([("a", b"1")]))
    r = publish(client, make_zip([("a", b"2")]))
    (tmp_path / "data" / "published" / b).unlink()
    resp = compare(client, b, r)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "artifact_missing" and err["package"] == "baseline"
    assert pair_of(client, b, r).status_code == 404


def test_not_a_zip_fails(tmp_path):
    client = make_client(tmp_path)
    b = publish(client, make_zip([("a", b"1")]))
    junk = publish(client, random.Random(5).randbytes(4096), "junk.bin")
    resp = compare(client, b, junk)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "bad_zip"
    assert pair_of(client, b, junk).status_code == 404


def test_duplicate_path_fails(tmp_path):
    client = make_client(tmp_path)
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # 故意制造重复路径
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("dup.txt", b"first")
            zf.writestr("ok.txt", b"ok")
            zf.writestr("dup.txt", b"second")  # 重复路径
    b = publish(client, make_zip([("x", b"1")]))
    r = publish(client, buf.getvalue(), "dup.zip")
    resp = compare(client, b, r)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "duplicate_path" and err["package"] == "review"
    assert pair_of(client, b, r).status_code == 404


def _set_encrypt_flag(zip_bytes: bytes) -> bytes:
    """Set the general-purpose encryption bit in local + central headers."""
    data = bytearray(zip_bytes)
    for sig, flag_off in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        pos = 0
        while True:
            pos = data.find(sig, pos)
            if pos < 0:
                break
            struct.pack_into("<H", data, pos + flag_off,
                             struct.unpack_from("<H", data, pos + flag_off)[0] | 0x1)
            pos += 4
    return bytes(data)


def test_encrypted_entry_fails(tmp_path):
    client = make_client(tmp_path)
    b = publish(client, make_zip([("x", b"1")]))
    r = publish(client, _set_encrypt_flag(make_zip([("secret.txt", b"s")])), "enc.zip")
    resp = compare(client, b, r)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "encrypted" and err["package"] == "review"
    assert pair_of(client, b, r).status_code == 404


def test_corrupt_entry_fails(tmp_path):
    client = make_client(tmp_path)
    good = make_zip([("data.bin", b"A" * 4096)])
    # 破坏压缩数据：定位 local header 之后的数据区并翻转一字节
    data = bytearray(good)
    assert data[:4] == b"PK\x03\x04"
    name_len, extra_len = struct.unpack_from("<HH", data, 26)
    start = 30 + name_len + extra_len
    data[start + 5] ^= 0xFF
    broken = bytes(data)

    b = publish(client, make_zip([("ok", b"1")]))
    r = publish(client, broken, "corrupt.zip")
    resp = compare(client, b, r)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "corrupt_entry" and err["package"] == "review"
    assert pair_of(client, b, r).status_code == 404


def test_too_many_entries_fails(tmp_path):
    client = make_client(tmp_path)
    many = make_zip([(f"f{i:05d}.txt", b"") for i in range(MAX_ENTRIES + 1)])
    b = publish(client, make_zip([("x", b"1")]))
    r = publish(client, many, "many.zip")
    resp = compare(client, b, r)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "too_many_entries"
    assert pair_of(client, b, r).status_code == 404


def test_too_large_fails(tmp_path):
    client = make_client(tmp_path)
    # 解压后 256 MiB + 1 字节（ deflate 后线上体积很小）
    huge = make_zip([("big.bin", b"\0" * (MAX_UNCOMPRESSED + 1))])
    assert len(huge) < 2 * 1024 * 1024
    b = publish(client, make_zip([("x", b"1")]))
    r = publish(client, huge, "huge.zip")
    resp = compare(client, b, r)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "too_large"
    assert pair_of(client, b, r).status_code == 404


def test_directory_entries_are_not_diffed(tmp_path):
    client = make_client(tmp_path)
    z1 = make_zip([("dir/", None), ("dir/a.txt", b"1")])
    z2 = make_zip([("dir/a.txt", b"1"), ("other/", None), ("other/b.txt", b"2")])
    b = publish(client, z1)
    r = publish(client, z2)
    body = compare(client, b, r).json()
    assert body["added"] == ["other/b.txt"]  # 目录条目本身不出现在差异里
    assert body["deleted"] == []
    assert body["changed"] == []


def test_empty_zips_compare_clean(tmp_path):
    client = make_client(tmp_path)
    b = publish(client, make_zip([]))
    r = publish(client, make_zip([]))
    resp = compare(client, b, r)
    assert resp.status_code == 201
    body = resp.json()
    assert body["added"] == body["deleted"] == body["changed"] == []
    assert body["unchanged_count"] == 0


def test_same_task_compared_with_itself(tmp_path):
    client = make_client(tmp_path)
    b = publish(client, make_zip([("a", b"1"), ("b", b"2")]))
    resp = compare(client, b, b)
    assert resp.status_code == 201
    body = resp.json()
    assert body["added"] == body["deleted"] == body["changed"] == []
    assert body["unchanged_count"] == 2


def test_record_survives_restart(tmp_path):
    client1 = make_client(tmp_path)
    b = publish(client1, make_zip(BASE_FILES), "baseline.zip")
    r = publish(client1, make_zip(REVIEW_FILES), "review.zip")
    first = compare(client1, b, r)
    assert first.status_code == 201

    # 进程重启：同一数据目录新建 app，记录仍可查
    client2 = make_client(tmp_path)
    got = pair_of(client2, b, r)
    assert got.status_code == 200
    assert got.json()["comparison_id"] == first.json()["comparison_id"]
    assert got.json()["changed"] == ["logs/sys.log"]
    assert len(client2.get("/api/comparisons").json()["comparisons"]) == 1


def test_concurrent_same_pair_leaves_single_record(tmp_path):
    client = make_client(tmp_path)
    b = publish(client, make_zip(BASE_FILES), "baseline.zip")
    r = publish(client, make_zip(REVIEW_FILES), "review.zip")

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda _: compare(client, b, r), range(6)))
    assert all(resp.status_code in (200, 201) for resp in results)
    ids = {resp.json()["comparison_id"] for resp in results}
    assert len(ids) == 1  # 所有并发请求拿到同一份记录
    lst = client.get("/api/comparisons").json()["comparisons"]
    assert len(lst) == 1


def test_concurrent_failure_leaves_no_record(tmp_path):
    client = make_client(tmp_path)
    b = publish(client, make_zip([("x", b"1")]))
    junk = publish(client, random.Random(6).randbytes(2048), "junk.bin")

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda _: compare(client, b, junk), range(4)))
    assert all(resp.status_code == 422 for resp in results)
    assert pair_of(client, b, junk).status_code == 404
    assert client.get("/api/comparisons").json()["comparisons"] == []


def test_validation_of_compare_request(tmp_path):
    client = make_client(tmp_path)
    assert client.post("/api/comparisons", json={}).status_code == 422
    assert client.post(
        "/api/comparisons", json={"baseline_id": "x"}).status_code == 422
    assert client.get("/api/comparisons/nope").status_code == 404
    assert pair_of(client, "a", "b").status_code == 404
