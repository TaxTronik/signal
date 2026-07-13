"""Freigegebenes llama.cpp-Release-Manifest für den nativen Engine-Download.

Diese Datei ist der unabhängige lokale Vertrauensanker. Sie darf nur im
kontrollierten Updateprozess aus dem README geändert werden; zur Laufzeit
werden weder ``latest`` noch Release-Metadaten abgefragt.
"""

from __future__ import annotations

from dataclasses import dataclass

LLAMA_CPP_RELEASE = "b9986"
LLAMA_CPP_RELEASE_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}"
)


@dataclass(frozen=True)
class EngineAsset:
    """Ein unabhängig im Repository gepinntes llama.cpp-Release-Artefakt."""

    name: str
    sha256: str
    groesse: int

    @property
    def url(self) -> str:
        # Die URL wird absichtlich aus dem festen Tag gebildet: kein Redirect
        # über /latest und keine zur Laufzeit abgefragte Release-Metadaten-URL.
        return f"{LLAMA_CPP_RELEASE_URL}/{self.name}"


# Quelle der Namen, Größen und Digests: offizielle GitHub-Release-API zu
# https://github.com/ggml-org/llama.cpp/releases/tag/b9986, Feld assets[].digest.
# Ermittelt und durch lokalen Download aller Assets gegengeprüft am 2026-07-13.
# Nur diese Windows-/Ubuntu-x64-Assets sind durch den Auto-Download unterstützt.
ENGINE_ASSETS: tuple[EngineAsset, ...] = (
    EngineAsset(
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
        391_443_627,
    ),
    EngineAsset(
        "cudart-llama-bin-win-cuda-13.3-x64.zip",
        "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e",
        390_970_417,
    ),
    EngineAsset(
        "llama-b9986-bin-ubuntu-rocm-7.2-x64.tar.gz",
        "e0cd217c318cb4b53561b5272f94371eeb8876ddc748f5cffc318cb97c64fe17",
        128_543_173,
    ),
    EngineAsset(
        "llama-b9986-bin-ubuntu-vulkan-x64.tar.gz",
        "6ec4c41dbb17590cf0dfbb21ccb842b8e235dc72d9e9556c1db9fe3bc390e768",
        31_206_236,
    ),
    EngineAsset(
        "llama-b9986-bin-ubuntu-x64.tar.gz",
        "0e94c66fabf6489569e67f0069ab34364b788bc4154763de2313a2318fe5c6cc",
        15_846_818,
    ),
    EngineAsset(
        "llama-b9986-bin-win-cpu-x64.zip",
        "df7b177d14697af9a1bd9a42e2d89455fc592e7206985ad1c672d19f3faa11d2",
        18_245_837,
    ),
    EngineAsset(
        "llama-b9986-bin-win-cuda-12.4-x64.zip",
        "0feaaac317856ac7c33131edb95851ef99731a7af97a7ee4a856b76a828ad1c2",
        267_384_331,
    ),
    EngineAsset(
        "llama-b9986-bin-win-cuda-13.3-x64.zip",
        "bc87be78e7cba5d07cdaf6f22dc99a6060e14ee5c89145f924db876e7461d25f",
        162_322_401,
    ),
    EngineAsset(
        "llama-b9986-bin-win-hip-radeon-x64.zip",
        "1ab5fa1e722cc766822b76cf6002e6b2a9c5d490d9e02579127f7ac62457be03",
        324_058_164,
    ),
    EngineAsset(
        "llama-b9986-bin-win-vulkan-x64.zip",
        "14f7e7bf66fa269b9f2950980a32851857290c60fae50e9dd44c13d144a5c2cd",
        32_934_678,
    ),
)
