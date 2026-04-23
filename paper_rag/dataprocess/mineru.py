from __future__ import annotations

import json
import urllib.error
import urllib.parse
import time
import http.client
import urllib.request
import zipfile
from pathlib import Path


class MinerUError(RuntimeError):
    pass


class MinerUClient:
    def __init__(self, api_key: str, base_url: str, model_version: str = "vlm", language: str = "en"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self.language = language

    def _request_json(self, method: str, url: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "*/*",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise MinerUError(f"MinerU {method} {url} failed: HTTP {exc.code}: {detail}") from exc
        if data.get("code") != 0:
            raise MinerUError(f"MinerU API failed: {data.get('msg') or data}")
        return data

    def parse_local_pdf(self, pdf_path: Path, output_dir: Path, data_id: str) -> Path:
        batch = self._create_upload_batch(pdf_path, data_id)
        upload_url = batch["file_urls"][0]
        self._upload_file(upload_url, pdf_path)
        result = self._wait_batch_result(batch["batch_id"], data_id)
        zip_url = result.get("full_zip_url")
        if not zip_url:
            raise MinerUError(f"MinerU result missing full_zip_url for {pdf_path.name}")
        return self._download_and_extract(zip_url, output_dir)

    def _create_upload_batch(self, pdf_path: Path, data_id: str) -> dict:
        payload = {
            "files": [{"name": pdf_path.name, "data_id": data_id}],
            "model_version": self.model_version,
            "language": self.language,
            "enable_formula": True,
            "enable_table": True,
        }
        data = self._request_json("POST", f"{self.base_url}/file-urls/batch", payload)
        return data["data"]

    def _upload_file(self, upload_url: str, pdf_path: Path) -> None:
        parsed = urllib.parse.urlparse(upload_url)
        if parsed.scheme != "https":
            raise MinerUError(f"Unsupported MinerU upload URL scheme: {parsed.scheme}")
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        headers = {"Content-Length": str(pdf_path.stat().st_size)}
        connection = http.client.HTTPSConnection(parsed.netloc, timeout=300)
        try:
            with pdf_path.open("rb") as handle:
                connection.request("PUT", path, body=handle, headers=headers, encode_chunked=False)
                response = connection.getresponse()
                detail = response.read().decode("utf-8", errors="replace").strip()
                if response.status >= 300:
                    raise MinerUError(f"MinerU upload failed for {pdf_path.name}: HTTP {response.status}: {detail}")
        finally:
            connection.close()

    def _wait_batch_result(self, batch_id: str, data_id: str, timeout_seconds: int = 1800) -> dict:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            data = self._request_json("GET", f"{self.base_url}/extract-results/batch/{batch_id}")
            results = data.get("data", {}).get("extract_result") or []
            if isinstance(results, dict):
                results = [results]
            for item in results:
                if item.get("data_id") not in (None, data_id):
                    continue
                state = item.get("state")
                if state == "done":
                    return item
                if state == "failed":
                    raise MinerUError(f"MinerU parse failed for {data_id}: {item}")
            time.sleep(10)
        raise MinerUError(f"Timed out waiting for MinerU batch {batch_id}")

    def _download_and_extract(self, zip_url: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / "_mineru_result.zip"
        try:
            with urllib.request.urlopen(zip_url, timeout=300) as response:
                zip_path.write_bytes(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise MinerUError(f"MinerU result download failed: HTTP {exc.code}: {detail}") from exc
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(output_dir)
        zip_path.unlink(missing_ok=True)
        return output_dir
