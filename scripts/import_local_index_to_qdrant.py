#!/usr/bin/env python3
"""Copy points from a local Qdrant database into a running Qdrant server.

The local and server storage layouts are different, so this script transfers
collection schema and points through the Qdrant client API instead of copying
storage files.

Run from the repository root, after stopping anything that has data/qdrant_db
open:

    uv run python scripts/import_local_index_to_qdrant.py \
      --local-db-path data/qdrant_db \
      --qdrant-url http://localhost:6333 \
      --collection egyptian_civil_code

By default, an existing destination collection is left untouched and causes an
error. Use --if-exists recreate only when you intentionally want to replace it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer a local Qdrant collection to a running Qdrant server."
    )
    parser.add_argument(
        "--local-db-path",
        default="data/qdrant_db",
        help="Path to local Qdrant database (default: data/qdrant_db).",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Destination Qdrant URL (default: QDRANT_URL or http://localhost:6333).",
    )
    parser.add_argument(
        "--collection",
        default="egyptian_civil_code",
        help="Collection name (default: egyptian_civil_code).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Points to transfer per batch (default: 128).",
    )
    parser.add_argument(
        "--if-exists",
        choices=("error", "skip", "recreate"),
        default="error",
        help=(
            "If destination collection exists: error (default), skip, or recreate. "
            "Recreate deletes the existing destination collection."
        ),
    )
    return parser.parse_args()


def create_destination_collection(
    destination: QdrantClient,
    collection_name: str,
    source_info: Any,
) -> None:
    """Create a destination collection using the source vector schema."""
    params = source_info.config.params
    vectors_config = getattr(params, "vectors", None)
    if vectors_config is None:
        raise RuntimeError(
            f"Source collection {collection_name!r} has no dense-vector configuration."
        )

    kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "vectors_config": vectors_config,
    }
    sparse_vectors_config = getattr(params, "sparse_vectors", None)
    if sparse_vectors_config:
        kwargs["sparse_vectors_config"] = sparse_vectors_config

    destination.create_collection(**kwargs)

    # Recreate payload indexes when the source client exposes their schema.
    # Points/payloads remain valid if an optional index cannot be recreated.
    payload_schema = getattr(source_info, "payload_schema", None) or {}
    for field_name, field_info in payload_schema.items():
        field_schema = getattr(field_info, "data_type", None)
        if field_schema is None:
            field_schema = getattr(field_info, "schema", None)
        if field_schema is None:
            continue
        try:
            destination.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        except (TypeError, ValueError, NotImplementedError) as exc:
            print(
                f"Warning: could not recreate payload index {field_name!r}: {exc}",
                file=sys.stderr,
            )


def transfer_collection(args: argparse.Namespace) -> None:
    local_path = Path(args.local_db_path).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Local Qdrant database does not exist: {local_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    # Keep source local mode separate from destination server mode.
    source = QdrantClient(path=str(local_path))
    destination = QdrantClient(
        url=args.qdrant_url,
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )

    try:
        # This also checks that the destination is reachable before changes are made.
        destination_collections = {
            item.name for item in destination.get_collections().collections
        }
        source_collections = {
            item.name for item in source.get_collections().collections
        }
        if args.collection not in source_collections:
            raise RuntimeError(
                f"Collection {args.collection!r} was not found in local database "
                f"{local_path}. Available collections: {sorted(source_collections)}"
            )

        source_info = source.get_collection(args.collection)
        if args.collection in destination_collections:
            if args.if_exists == "skip":
                print(
                    f"Destination collection {args.collection!r} already exists; "
                    "skipping import."
                )
                return
            if args.if_exists == "error":
                raise RuntimeError(
                    f"Destination collection {args.collection!r} already exists on "
                    f"{args.qdrant_url}. Nothing was changed. To intentionally replace "
                    "it, rerun with --if-exists recreate."
                )
            print(
                f"Replacing destination collection {args.collection!r} because "
                "--if-exists=recreate was specified."
            )
            destination.delete_collection(collection_name=args.collection)

        total_source = source.count(
            collection_name=args.collection,
            exact=True,
        ).count
        create_destination_collection(destination, args.collection, source_info)
        print(
            f"Transferring {total_source} point(s) from {local_path} to "
            f"{args.qdrant_url}; collection={args.collection!r}."
        )

        transferred = 0
        offset = None
        while True:
            records, next_offset = source.scroll(
                collection_name=args.collection,
                limit=args.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                break

            points = [
                models.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload or {},
                )
                for record in records
            ]
            destination.upsert(
                collection_name=args.collection,
                points=points,
                wait=True,
            )
            transferred += len(points)
            print(f"Transferred {transferred}/{total_source}", flush=True)

            if next_offset is None:
                break
            offset = next_offset

        total_destination = destination.count(
            collection_name=args.collection,
            exact=True,
        ).count
        if total_destination != total_source:
            raise RuntimeError(
                "Point-count verification failed: "
                f"local={total_source}, destination={total_destination}. "
                "Check the destination collection before retrying."
            )

        print(
            f"Success: verified {total_destination} point(s) in destination "
            f"collection {args.collection!r}."
        )
    finally:
        source.close()
        destination.close()


def main() -> int:
    args = parse_args()
    try:
        transfer_collection(args)
    except Exception as exc: # noqa: BLE001 — CLI boundary for clear error reporting
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
