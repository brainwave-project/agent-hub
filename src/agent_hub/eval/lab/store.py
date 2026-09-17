"""Versioned, hash-addressed file store for datasets, experiments and runs.

Plain files rather than a database, on purpose: a golden set is an artefact you
review in a diff, and a frozen version should be something you can commit.

Layout under the store root (`EVAL_LAB_DIR`):

    datasets/<id>/dataset.json              name, description
    datasets/<id>/v<N>/manifest.json        status, hash, split config
    datasets/<id>/v<N>/corpus/<doc>.json    one document
    datasets/<id>/v<N>/golden.jsonl         golden items, one per line
    datasets/<id>/v<N>/indexes.json         index state (operational, not content)
    datasets/<id>/v<N>/pool.json            pooled candidates for judging     } artefacts:
    datasets/<id>/v<N>/annotations.json     blind re-annotations (label audit) } about the
    datasets/<id>/v<N>/candidates.json      synthetic questions under review   } content,
    experiments/<id>.json                                                        not part
    runs/<id>.json                                                               of its hash
    run_annotations/<run_id>.json           failure-analysis tags
    learning.json                           Learning Path progress

Immutability rule: once a version is frozen, its corpus, items and split can no
longer be written. The content hash is recorded at freeze time and recomputed on
every readiness check, so an out-of-band edit to a frozen file is detected
rather than silently changing what "v3" means. Artefacts stay writable on a
frozen version: auditing a frozen set's labels does not change the set.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import threading
import uuid
from dataclasses import dataclass

from agent_hub.eval.lab.models import (
    CorpusDoc,
    DatasetManifest,
    DatasetMeta,
    Experiment,
    GoldenItem,
    IndexState,
    Run,
    now_iso,
)

__all__ = [
    "LabStore",
    "Bundle",
    "NotFound",
    "Immutable",
    "Conflict",
    "ARTIFACTS",
    "content_hash",
    "corpus_hash",
    "owner_id_for",
]

#: Namespace for deterministic owner ids. Changing it orphans every index.
_OWNER_NAMESPACE = uuid.UUID("5b2c7e0a-3f1d-4c8e-9a61-0e7a1ab5e7a1")

#: Per-version files that describe work ON the content without being content.
ARTIFACTS = ("pool", "annotations", "candidates")

#: Item fields added after the first schema. They are left out of the hash while
#: they hold their default, so adding a field never changes the hash of a
#: dataset frozen before the field existed — which would read as tampering.
_ITEM_FIELDS_ADDED_LATER: dict[str, object] = {"judged_nonrelevant": [], "origin": "human"}

_LOCK = threading.RLock()


class NotFound(LookupError):
    pass


class Immutable(PermissionError):
    pass


class Conflict(ValueError):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _item_payload(item: GoldenItem) -> dict:
    data = item.model_dump(mode="json")
    for name, default in _ITEM_FIELDS_ADDED_LATER.items():
        if data.get(name) == default:
            data.pop(name, None)
    return data


def content_hash(corpus: dict[str, CorpusDoc], items: list[GoldenItem]) -> str:
    """Hash of everything a score depends on: documents, questions, labels, split."""
    payload = {
        "corpus": sorted([d.doc_id, d.lang, _sha(d.body)] for d in corpus.values()),
        "items": [_item_payload(i) for i in sorted(items, key=lambda i: i.id)],
    }
    return _sha(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def corpus_hash(corpus: dict[str, CorpusDoc], normalization: str) -> str:
    """Hash of what an index is built from. Questions are deliberately excluded."""
    payload = {
        "corpus": sorted([d.doc_id, _sha(d.body)] for d in corpus.values()),
        "normalization": normalization,
    }
    return _sha(json.dumps(payload, sort_keys=True))


def owner_id_for(dataset_id: str, corpus_digest: str) -> str:
    """Synthetic know-hub owner for one (dataset, corpus content) pair.

    A strictly valid UUID (know-hub casts it to ::uuid). Deterministic, so
    re-indexing an unchanged corpus finds its own earlier uploads, and a changed
    corpus lands in a fresh, isolated namespace instead of mixing old and new
    document versions in one ranking.
    """
    return str(uuid.uuid5(_OWNER_NAMESPACE, f"{dataset_id}:{corpus_digest}"))


def _write_json(path: pathlib.Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # atomic on the same filesystem


def _read_json(path: pathlib.Path) -> dict:
    if not path.exists():
        raise NotFound(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Bundle:
    meta: DatasetMeta
    manifest: DatasetManifest
    corpus: dict[str, CorpusDoc]
    items: list[GoldenItem]
    indexes: dict[str, IndexState]

    @property
    def current_hash(self) -> str:
        return content_hash(self.corpus, self.items)

    def corpus_hash(self, normalization: str) -> str:
        return corpus_hash(self.corpus, normalization)

    def item(self, item_id: str) -> GoldenItem | None:
        return next((i for i in self.items if i.id == item_id), None)


class LabStore:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        for sub in ("datasets", "experiments", "runs", "run_annotations"):
            (root / sub).mkdir(parents=True, exist_ok=True)

    # --- paths --------------------------------------------------------------

    def _ds(self, dataset_id: str) -> pathlib.Path:
        return self.root / "datasets" / dataset_id

    def _ver(self, dataset_id: str, version: int) -> pathlib.Path:
        path = self._ds(dataset_id) / f"v{version}"
        if not path.exists():
            raise NotFound(f"dataset {dataset_id} v{version}")
        return path

    # --- datasets -----------------------------------------------------------

    def versions(self, dataset_id: str) -> list[int]:
        directory = self._ds(dataset_id)
        if not directory.exists():
            raise NotFound(f"dataset {dataset_id}")
        return sorted(
            int(p.name[1:]) for p in directory.iterdir() if p.is_dir() and p.name[1:].isdigit()
        )

    def list_datasets(self) -> list[dict]:
        out = []
        for directory in sorted((self.root / "datasets").iterdir()):
            if not (directory / "dataset.json").exists():
                continue
            meta = self.get_meta(directory.name)
            versions = []
            for v in self.versions(directory.name):
                manifest = self.manifest(directory.name, v)
                vdir = self._ver(directory.name, v)
                versions.append(
                    {
                        **manifest.model_dump(),
                        "items": len(self._read_items(vdir)),
                        "documents": len(list((vdir / "corpus").glob("*.json"))),
                    }
                )
            out.append({**meta.model_dump(), "versions": versions})
        return out

    def get_meta(self, dataset_id: str) -> DatasetMeta:
        return DatasetMeta.model_validate(_read_json(self._ds(dataset_id) / "dataset.json"))

    def create_dataset(self, meta: DatasetMeta) -> DatasetManifest:
        with _LOCK:
            if (self._ds(meta.id) / "dataset.json").exists():
                raise Conflict(f"dataset {meta.id} already exists")
            _write_json(self._ds(meta.id) / "dataset.json", meta.model_dump())
            manifest = DatasetManifest(dataset_id=meta.id, version=1)
            vdir = self._ds(meta.id) / "v1"
            (vdir / "corpus").mkdir(parents=True, exist_ok=True)
            (vdir / "golden.jsonl").write_text("", encoding="utf-8")
            _write_json(vdir / "manifest.json", manifest.model_dump())
            return manifest

    def update_meta(
        self, dataset_id: str, name: str | None, description: str | None
    ) -> DatasetMeta:
        with _LOCK:
            meta = self.get_meta(dataset_id)
            if name is not None:
                meta.name = name
            if description is not None:
                meta.description = description
            _write_json(self._ds(dataset_id) / "dataset.json", meta.model_dump())
            return meta

    def manifest(self, dataset_id: str, version: int) -> DatasetManifest:
        return DatasetManifest.model_validate(
            _read_json(self._ver(dataset_id, version) / "manifest.json")
        )

    def _save_manifest(self, manifest: DatasetManifest) -> None:
        _write_json(
            self._ver(manifest.dataset_id, manifest.version) / "manifest.json",
            manifest.model_dump(),
        )

    def _read_items(self, vdir: pathlib.Path) -> list[GoldenItem]:
        path = vdir / "golden.jsonl"
        if not path.exists():
            return []
        return [
            GoldenItem.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def corpus(self, dataset_id: str, version: int) -> dict[str, CorpusDoc]:
        vdir = self._ver(dataset_id, version)
        docs = [
            CorpusDoc.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted((vdir / "corpus").glob("*.json"))
        ]
        return {d.doc_id: d for d in docs}

    def items(self, dataset_id: str, version: int) -> list[GoldenItem]:
        return self._read_items(self._ver(dataset_id, version))

    def indexes(self, dataset_id: str, version: int) -> dict[str, IndexState]:
        path = self._ver(dataset_id, version) / "indexes.json"
        if not path.exists():
            return {}
        return {k: IndexState.model_validate(v) for k, v in _read_json(path).items()}

    def save_index(self, dataset_id: str, version: int, state: IndexState) -> None:
        with _LOCK:
            current = self.indexes(dataset_id, version)
            current[state.normalization] = state
            _write_json(
                self._ver(dataset_id, version) / "indexes.json",
                {k: v.model_dump() for k, v in current.items()},
            )

    def bundle(self, dataset_id: str, version: int) -> Bundle:
        return Bundle(
            meta=self.get_meta(dataset_id),
            manifest=self.manifest(dataset_id, version),
            corpus=self.corpus(dataset_id, version),
            items=self.items(dataset_id, version),
            indexes=self.indexes(dataset_id, version),
        )

    # --- artefacts (writable on frozen versions) --------------------------------

    def artifact(self, dataset_id: str, version: int, name: str, default: object = None):
        if name not in ARTIFACTS:
            raise ValueError(f"unknown artefact {name!r}")
        path = self._ver(dataset_id, version) / f"{name}.json"
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def save_artifact(self, dataset_id: str, version: int, name: str, data: object) -> None:
        if name not in ARTIFACTS:
            raise ValueError(f"unknown artefact {name!r}")
        with _LOCK:
            _write_json(self._ver(dataset_id, version) / f"{name}.json", data)

    def run_annotations(self, run_id: str) -> dict:
        path = self.root / "run_annotations" / f"{run_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save_run_annotations(self, run_id: str, data: dict) -> None:
        with _LOCK:
            _write_json(self.root / "run_annotations" / f"{run_id}.json", data)

    def all_run_annotations(self) -> dict[str, dict]:
        return {
            p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in (self.root / "run_annotations").glob("*.json")
        }

    def learning_progress(self) -> dict:
        path = self.root / "learning.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save_learning_progress(self, data: dict) -> None:
        with _LOCK:
            _write_json(self.root / "learning.json", data)

    # --- writes (drafts only) -----------------------------------------------

    def _require_draft(self, dataset_id: str, version: int) -> DatasetManifest:
        manifest = self.manifest(dataset_id, version)
        if manifest.status == "frozen":
            raise Immutable(f"{dataset_id} v{version} is frozen — fork a new version to change it")
        return manifest

    def put_doc(self, dataset_id: str, version: int, doc: CorpusDoc) -> None:
        with _LOCK:
            self._require_draft(dataset_id, version)
            _write_json(
                self._ver(dataset_id, version) / "corpus" / f"{doc.doc_id}.json", doc.model_dump()
            )

    def delete_doc(self, dataset_id: str, version: int, doc_id: str) -> None:
        with _LOCK:
            self._require_draft(dataset_id, version)
            path = self._ver(dataset_id, version) / "corpus" / f"{doc_id}.json"
            if not path.exists():
                raise NotFound(f"document {doc_id}")
            path.unlink()

    def replace_items(self, dataset_id: str, version: int, items: list[GoldenItem]) -> None:
        with _LOCK:
            self._require_draft(dataset_id, version)
            ids = [i.id for i in items]
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                raise Conflict(f"duplicate item ids: {', '.join(dupes)}")
            path = self._ver(dataset_id, version) / "golden.jsonl"
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
            os.replace(tmp, path)

    def put_item(self, dataset_id: str, version: int, item: GoldenItem) -> None:
        with _LOCK:
            existing = self.items(dataset_id, version)
            items = [i for i in existing if i.id != item.id]
            position = next((n for n, i in enumerate(existing) if i.id == item.id), len(items))
            items.insert(position, item)
            self.replace_items(dataset_id, version, items)

    def delete_item(self, dataset_id: str, version: int, item_id: str) -> None:
        with _LOCK:
            items = self.items(dataset_id, version)
            remaining = [i for i in items if i.id != item_id]
            if len(remaining) == len(items):
                raise NotFound(f"item {item_id}")
            self.replace_items(dataset_id, version, remaining)

    def set_split(
        self, dataset_id: str, version: int, items: list[GoldenItem], manifest: DatasetManifest
    ) -> None:
        with _LOCK:
            self._require_draft(dataset_id, version)
            self.replace_items(dataset_id, version, items)
            manifest.split_assigned_at = now_iso()
            self._save_manifest(manifest)

    def set_changelog(self, dataset_id: str, version: int, changelog: str) -> DatasetManifest:
        with _LOCK:
            manifest = self._require_draft(dataset_id, version)
            manifest.changelog = changelog
            self._save_manifest(manifest)
            return manifest

    # --- lifecycle ------------------------------------------------------------

    def freeze(self, dataset_id: str, version: int) -> DatasetManifest:
        with _LOCK:
            manifest = self._require_draft(dataset_id, version)
            bundle = self.bundle(dataset_id, version)
            if not bundle.corpus or not bundle.items:
                raise Conflict("cannot freeze an empty dataset — add documents and items first")
            manifest.status = "frozen"
            manifest.frozen_at = now_iso()
            manifest.content_hash = bundle.current_hash
            self._save_manifest(manifest)
            return manifest

    def fork(self, dataset_id: str, version: int, changelog: str = "") -> DatasetManifest:
        with _LOCK:
            source = self._ver(dataset_id, version)
            parent = self.manifest(dataset_id, version)
            new_version = max(self.versions(dataset_id)) + 1
            target = self._ds(dataset_id) / f"v{new_version}"
            shutil.copytree(source, target)
            manifest = DatasetManifest(
                dataset_id=dataset_id,
                version=new_version,
                parent_version=version,
                split_config=parent.split_config,
                split_assigned_at=parent.split_assigned_at,
                changelog=changelog,
            )
            _write_json(target / "manifest.json", manifest.model_dump())
            return manifest

    # --- experiments ----------------------------------------------------------

    def list_experiments(self) -> list[Experiment]:
        return [
            Experiment.model_validate(_read_json(p))
            for p in sorted((self.root / "experiments").glob("*.json"))
        ]

    def get_experiment(self, experiment_id: str) -> Experiment:
        return Experiment.model_validate(
            _read_json(self.root / "experiments" / f"{experiment_id}.json")
        )

    def save_experiment(self, experiment: Experiment) -> None:
        with _LOCK:
            _write_json(
                self.root / "experiments" / f"{experiment.id}.json", experiment.model_dump()
            )

    def delete_experiment(self, experiment_id: str) -> None:
        with _LOCK:
            experiment = self.get_experiment(experiment_id)
            if experiment.status != "draft":
                raise Immutable(
                    "only draft experiments can be deleted — registered ones are a record"
                )
            (self.root / "experiments" / f"{experiment_id}.json").unlink()

    # --- runs -----------------------------------------------------------------

    def save_run(self, run: Run) -> None:
        with _LOCK:
            _write_json(self.root / "runs" / f"{run.id}.json", run.model_dump())

    def get_run(self, run_id: str) -> Run:
        return Run.model_validate(_read_json(self.root / "runs" / f"{run_id}.json"))

    def list_runs(self, dataset_id: str | None = None) -> list[dict]:
        """Run summaries (no per-query results), newest first."""
        out = []
        for path in (self.root / "runs").glob("*.json"):
            raw = _read_json(path)
            if dataset_id and raw.get("dataset_id") != dataset_id:
                continue
            raw.pop("results", None)
            out.append(raw)
        return sorted(out, key=lambda r: r.get("created_at", ""), reverse=True)
