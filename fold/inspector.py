"""
Inspector module for Sheprd.
Parses GGUF models directly, discovers hardware topology (CPU, RAM, GPU/Vulkan),
and computes optimal llama-server execution parameters.
"""

import ctypes.util
import glob
import io
import os
import re
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .security import SecurityError, validate_model_path

# Mapping GGUF file_type integer enum to human-readable quantization string
GGUF_FILE_TYPE_NAMES = {
    0: "ALL_F32",
    1: "MOSTLY_F16",
    2: "MOSTLY_Q4_0",
    3: "MOSTLY_Q4_1",
    7: "MOSTLY_Q8_0",
    8: "MOSTLY_Q5_0",
    9: "MOSTLY_Q5_1",
    10: "MOSTLY_Q2_K",
    11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M",
    13: "MOSTLY_Q3_K_L",
    14: "MOSTLY_Q4_K_S",
    15: "MOSTLY_Q4_K_M",
    16: "MOSTLY_Q5_K_S",
    17: "MOSTLY_Q5_K_M",
    18: "MOSTLY_Q6_K",
    19: "MOSTLY_IQ2_XXS",
    20: "MOSTLY_IQ2_XS",
    21: "MOSTLY_Q2_K_S",
    22: "MOSTLY_IQ3_XS",
    23: "MOSTLY_IQ3_XXS",
    24: "MOSTLY_IQ1_S",
    25: "MOSTLY_IQ4_NL",
    26: "MOSTLY_IQ3_S",
    27: "MOSTLY_IQ3_M",
    28: "MOSTLY_IQ2_S",
    29: "MOSTLY_IQ2_M",
    30: "MOSTLY_IQ4_XS",
    31: "MOSTLY_IQ1_M",
    32: "MOSTLY_BF16",
    33: "MOSTLY_Q4_0_4_4",
    34: "MOSTLY_Q4_0_4_8",
    35: "MOSTLY_Q4_0_8_8",
    36: "MOSTLY_TQ1_0",
    37: "MOSTLY_TQ2_0",
    38: "MOSTLY_MXFP4",
}


def _extract_int(val: Any, default: int = 0) -> int:
    """Safely extracts an integer from numbers, numeric strings, or collections of numbers."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, (list, tuple, set)):
        numbers = []
        for x in val:
            try:
                numbers.append(int(x))
            except (ValueError, TypeError):
                continue
        return max(numbers) if numbers else default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


@dataclass
class HardwareProfile:
    cpu_model: str
    physical_cores: int
    logical_cores: int
    total_ram_bytes: int
    available_ram_bytes: int
    gpu_devices: List[str]
    vulkan_supported: bool
    recommended_threads: int


@dataclass
class ModelMetadata:
    file_path: str
    file_name: str
    file_size_bytes: int
    file_size_gb: float
    architecture: str
    model_name: str
    quantization: str
    layer_count: int
    context_length: int
    embedding_length: int
    head_count: int
    head_count_kv: int
    chat_template_raw: Optional[str]
    detected_template_kind: str
    estimated_vram_mb: int


@dataclass
class ServerConfigRecommendation:
    model_path: str
    architecture: str
    context_size: int
    n_gpu_layers: int
    threads: int
    batch_size: int
    ubatch_size: int
    template_kind: str
    flash_attention: bool
    recommended_port: int
    summary_notes: List[str]


class GGUFInspector:
    """Reads GGUF metadata quickly without loading model tensors into memory."""

    @staticmethod
    def _read_str(stream: io.BufferedReader) -> str:
        length_bytes = stream.read(8)
        if len(length_bytes) < 8:
            raise EOFError("Unexpected EOF while reading string length")
        length = struct.unpack("<Q", length_bytes)[0]
        # Safety bound: reject unreasonably huge strings (> 5MB)
        if length > 5 * 1024 * 1024:
            raise ValueError(f"String length {length} exceeds maximum safety bound.")
        val_bytes = stream.read(length)
        if len(val_bytes) < length:
            raise EOFError("Unexpected EOF while reading string content")
        return val_bytes.decode("utf-8", errors="replace")

    @classmethod
    def _read_val(cls, stream: io.BufferedReader, vtype: int, key_name: str = "") -> Any:
        if vtype == 0:  # UINT8
            return struct.unpack("<B", stream.read(1))[0]
        elif vtype == 1:  # INT8
            return struct.unpack("<b", stream.read(1))[0]
        elif vtype == 2:  # UINT16
            return struct.unpack("<H", stream.read(2))[0]
        elif vtype == 3:  # INT16
            return struct.unpack("<h", stream.read(2))[0]
        elif vtype == 4:  # UINT32
            return struct.unpack("<I", stream.read(4))[0]
        elif vtype == 5:  # INT32
            return struct.unpack("<i", stream.read(4))[0]
        elif vtype == 6:  # FLOAT32
            return struct.unpack("<f", stream.read(4))[0]
        elif vtype == 7:  # BOOL
            return bool(struct.unpack("<B", stream.read(1))[0])
        elif vtype == 8:  # STRING
            return cls._read_str(stream)
        elif vtype == 9:  # ARRAY
            header = stream.read(12)
            if len(header) < 12:
                raise EOFError("Unexpected EOF reading array header")
            item_type, count = struct.unpack("<IQ", header)
            # Skip large token lists (e.g. tokenizer vocabularies) for speed
            if key_name.startswith("tokenizer.ggml.tokens") or count > 500:
                # Calculate skip size if fixed-width, or read strings iteratively to seek past
                if item_type == 8:  # Strings
                    for _ in range(count):
                        l = struct.unpack("<Q", stream.read(8))[0]
                        stream.seek(l, io.SEEK_CUR)
                elif item_type in (0, 1, 7):
                    stream.seek(count * 1, io.SEEK_CUR)
                elif item_type in (2, 3):
                    stream.seek(count * 2, io.SEEK_CUR)
                elif item_type in (4, 5, 6):
                    stream.seek(count * 4, io.SEEK_CUR)
                elif item_type in (10, 11, 12):
                    stream.seek(count * 8, io.SEEK_CUR)
                return f"[Array of {count} items skipped]"
            return [cls._read_val(stream, item_type, key_name) for _ in range(count)]
        elif vtype == 10:  # UINT64
            return struct.unpack("<Q", stream.read(8))[0]
        elif vtype == 11:  # INT64
            return struct.unpack("<q", stream.read(8))[0]
        elif vtype == 12:  # FLOAT64
            return struct.unpack("<d", stream.read(8))[0]
        else:
            raise ValueError(f"Unknown GGUF value type {vtype}")

    @classmethod
    def inspect(cls, file_path: Path) -> ModelMetadata:
        target = validate_model_path(str(file_path))
        file_size = target.stat().st_size

        # Multi-part / Sharded GGUF detection (e.g., -00001-of-00005.gguf)
        shard_match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", target.name, re.IGNORECASE)
        if shard_match:
            base_pattern = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "-*-of-*.gguf", target.name)
            shard_files = sorted(target.parent.glob(base_pattern))
            if shard_files:
                file_size = sum(f.stat().st_size for f in shard_files)

        kv_data: Dict[str, Any] = {}

        with open(target, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise SecurityError("Invalid GGUF header magic bytes.")

            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version {version}")

            tensors_count, kv_count = struct.unpack("<QQ", f.read(16))

            for _ in range(kv_count):
                key = cls._read_str(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                val = cls._read_val(f, vtype, key_name=key)
                kv_data[key] = val

        # Extract architecture and model properties
        arch = str(kv_data.get("general.architecture", "llama"))
        model_name = str(
            kv_data.get("general.name")
            or kv_data.get("general.basename")
            or target.stem
        )

        file_type_code = kv_data.get("general.file_type")
        quantization = GGUF_FILE_TYPE_NAMES.get(file_type_code) if file_type_code is not None else None
        if not quantization or quantization.startswith("TYPE_"):
            # Fallback to pattern matching from filename (e.g. Q4_K_M, IQ4_NL, UD-Q4_K_XL, MXFP4)
            quant_match = re.search(
                r"(UD-[A-Z0-9_]+|IQ[1-4]_[A-Z0-9_]+|Q[2-8]_[A-Z0-9_]+|MXFP[0-9_A-Z]+|BF16|F16|F32)",
                target.name,
                re.IGNORECASE,
            )
            if quant_match:
                quantization = quant_match.group(1).upper()
            elif file_type_code is not None:
                quantization = f"TYPE_{file_type_code}"
            else:
                quantization = "UNKNOWN"

        # Layer count / block count
        layer_count = _extract_int(
            kv_data.get(f"{arch}.block_count")
            or kv_data.get("block_count"),
            default=32,
        )

        # Context length
        context_len = _extract_int(
            kv_data.get(f"{arch}.context_length")
            or kv_data.get("context_length"),
            default=8192,
        )

        embedding_len = _extract_int(
            kv_data.get(f"{arch}.embedding_length")
            or kv_data.get("embedding_length"),
            default=4096,
        )

        head_count = _extract_int(
            kv_data.get(f"{arch}.attention.head_count")
            or kv_data.get("head_count"),
            default=32,
        )

        # Handle architectures like Gemma 4 where head_count_kv is a list of ints per layer
        raw_head_count_kv = (
            kv_data.get(f"{arch}.attention.head_count_kv")
            or kv_data.get("head_count_kv")
        )
        if isinstance(raw_head_count_kv, (list, tuple, set)):
            valid_nums = []
            for x in raw_head_count_kv:
                try:
                    valid_nums.append(int(x))
                except (ValueError, TypeError):
                    pass
            head_count_kv = max(valid_nums) if valid_nums else head_count
            total_kv_heads = sum(valid_nums) if valid_nums else (layer_count * head_count_kv)
        else:
            head_count_kv = _extract_int(raw_head_count_kv, default=head_count)
            total_kv_heads = layer_count * head_count_kv

        chat_template = kv_data.get("tokenizer.chat_template")
        if not isinstance(chat_template, str):
            chat_template = None

        template_kind = cls.detect_template_kind(chat_template, arch, model_name)

        # Calculate estimated VRAM requirement (weights + KV cache buffer)
        weights_mb = int(file_size / (1024 * 1024))
        head_dim = embedding_len // max(1, head_count)
        # KV cache estimate: 2 (K+V) * total_kv_heads * head_dim * bytes_per_elem (fp16 = 2)
        kv_bytes_per_token = 2 * total_kv_heads * head_dim * 2
        est_ctx = min(context_len, 8192)
        kv_cache_mb = int((kv_bytes_per_token * est_ctx) / (1024 * 1024))
        est_vram_mb = weights_mb + kv_cache_mb + 256  # 256MB overhead

        return ModelMetadata(
            file_path=str(target),
            file_name=target.name,
            file_size_bytes=file_size,
            file_size_gb=round(file_size / (1024 ** 3), 2),
            architecture=arch,
            model_name=model_name,
            quantization=quantization,
            layer_count=layer_count,
            context_length=context_len,
            embedding_length=embedding_len,
            head_count=head_count,
            head_count_kv=head_count_kv,
            chat_template_raw=chat_template,
            detected_template_kind=template_kind,
            estimated_vram_mb=est_vram_mb,
        )

    @staticmethod
    def detect_template_kind(chat_template: Optional[str], arch: str, model_name: str) -> str:
        """Determines the standard chat template syntax expected by the model."""
        content = (chat_template or "").lower()
        name = model_name.lower()
        arch = arch.lower()

        if "<|im_start|>" in content or "chatml" in content or "qwen" in name or "qwen" in arch:
            return "chatml"
        elif "<|start_header_id|>" in content or "llama-3" in name or "llama3" in name:
            return "llama-3"
        elif "[inst]" in content or "mistral" in name or "mixtral" in name or "devstral" in name:
            return "mistral"
        elif "<start_of_turn>" in content or "gemma" in name or "gemma" in arch:
            return "gemma"
        elif "<|user|>" in content or "phi-3" in name or "phi3" in name or "phi" in arch:
            return "phi-3"
        elif "deepseek" in name or "<｜user｜>" in content or "deepseek" in arch:
            return "deepseek"
        elif "glm" in name or "glm" in arch:
            return "chatml"
        return "chatml"  # Universal modern fallback for instruct models


class HardwareInspector:
    """Discovers host CPU, memory, and GPU/Vulkan capabilities."""

    @staticmethod
    def detect() -> HardwareProfile:
        cpu_model = "Unknown CPU"
        physical_cores = 8
        logical_cores = os.cpu_count() or 8

        # Read /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

        # Memory detection
        total_ram = 16 * 1024 * 1024 * 1024
        avail_ram = 8 * 1024 * 1024 * 1024
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_ram = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        avail_ram = int(line.split()[1]) * 1024
        except Exception:
            pass

        # GPU detection via lspci
        gpu_devices = []
        try:
            lspci_out = subprocess.run(
                ["lspci"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=3,
            ).stdout
            for line in lspci_out.splitlines():
                if any(x in line.lower() for x in ["vga", "3d", "display"]):
                    gpu_devices.append(line.split(":", 2)[-1].strip())
        except Exception:
            pass

        # Vulkan check: check find_library, multiarch, and standard library paths (B12/N16)
        vulkan_supported = bool(ctypes.util.find_library("vulkan"))
        if not vulkan_supported:
            vulkan_candidates = [
                "/usr/lib/libvulkan.so*",
                "/usr/lib64/libvulkan.so*",
                "/usr/lib/*-linux-gnu/libvulkan.so*",
                "/usr/local/lib/libvulkan.so*",
            ]
            for pattern in vulkan_candidates:
                if glob.glob(pattern):
                    vulkan_supported = True
                    break

        # Optimal threads: use number of physical cores (typically logical_cores // 2 or clamped to 8)
        rec_threads = min(8, max(2, logical_cores // 2 if logical_cores > 4 else logical_cores))

        return HardwareProfile(
            cpu_model=cpu_model,
            physical_cores=max(1, logical_cores // 2),
            logical_cores=logical_cores,
            total_ram_bytes=total_ram,
            available_ram_bytes=avail_ram,
            gpu_devices=gpu_devices,
            vulkan_supported=vulkan_supported,
            recommended_threads=rec_threads,
        )


def analyze_model_and_recommend(
    model_path_str: str,
    target_port: int = 8081,
) -> Tuple[ModelMetadata, ServerConfigRecommendation]:
    """
    Main analysis pipeline:
    Validates model, extracts GGUF metadata, detects hardware,
    and returns complete auto-calculated execution configuration.
    """
    path = Path(model_path_str)
    meta = GGUFInspector.inspect(path)
    hw = HardwareInspector.detect()

    notes = []

    # Calculate optimal context length
    # Native model context can be huge (e.g. 32k, 128k). Default to a snappy, memory-efficient 8192 or native if smaller.
    if meta.context_length <= 8192:
        ctx_size = meta.context_length
    else:
        ctx_size = 8192  # Balanced context for local speed and VRAM economy
        notes.append(f"Model supports up to {meta.context_length} context; defaulted to 8192 for optimal speed.")

    # Calculate GPU offload layers (--n-gpu-layers)
    # Check if GPU is present and Vulkan supported
    has_discrete_gpu = any("amd" in d.lower() or "nvidia" in d.lower() or "radeon" in d.lower() for d in hw.gpu_devices)
    if hw.vulkan_supported and has_discrete_gpu:
        # Offload all layers to GPU via Vulkan for maximum inference speed
        n_gpu_layers = meta.layer_count + 1
        notes.append(f"Detected {hw.gpu_devices[0] if hw.gpu_devices else 'Discrete GPU'} with Vulkan: offloading all {meta.layer_count} layers.")
    else:
        n_gpu_layers = 0
        notes.append(f"No discrete GPU offload detected; configured for multi-threaded CPU inference ({hw.recommended_threads} threads).")

    # Threads
    threads = hw.recommended_threads

    # Sharded multi-part model notes
    if "-00001-of-" in path.name:
        shard_match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path.name, re.IGNORECASE)
        if shard_match:
            total_shards = int(shard_match.group(2))
            notes.append(f"Sharded model: {total_shards} parts detected (total size {meta.file_size_gb} GB). llama.cpp will automatically load all shards.")

    # Large model notes
    if meta.file_size_gb > 25:
        notes.append(f"Large model ({meta.file_size_gb} GB): high-capacity model; extended startup timeout allocated.")

    # Gemma architecture notes
    if "gemma" in meta.architecture.lower() or "gemma" in meta.model_name.lower():
        notes.append("Gemma architecture: configured with Gemma chat template and sliding-window KV attention.")

    # Batch sizes
    batch_size = 2048
    ubatch_size = 512

    rec = ServerConfigRecommendation(
        model_path=str(path.resolve()),
        architecture=meta.architecture,
        context_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        threads=threads,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        template_kind=meta.detected_template_kind,
        flash_attention=True,
        recommended_port=target_port,
        summary_notes=notes,
    )

    return meta, rec
