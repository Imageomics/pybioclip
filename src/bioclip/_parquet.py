"""Optional Parquet serialization for the embed command."""

import json
import os
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


def write_embeddings_parquet(embeddings: dict[str, list[float]], output: str,
                             model: str, pretrained: str | None) -> None:
    if len({len(vector) for vector in embeddings.values()}) > 1:
        raise ValueError("All embeddings must have the same dimensionality")
    metadata = {
        b'pybioclip': json.dumps({
            'model': model,
            'pretrained': pretrained,
            'normalized': False,
        }).encode('utf-8')
    }
    schema = pa.schema([
        pa.field('uuid', pa.string()),
        pa.field('emb', pa.list_(pa.float32())),
    ], metadata=metadata)
    table = pa.Table.from_arrays([
        pa.array(list(embeddings), type=pa.string()),
        pa.array(list(embeddings.values()), type=pa.list_(pa.float32())),
    ], schema=schema)

    output_path = Path(output)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as directory:
        temporary_path = Path(directory) / 'embeddings.parquet'
        pq.write_table(table, temporary_path)
        os.replace(temporary_path, output_path)
