"""Available methods endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

from splattie.methods.registry import registry
from splattie.types import MethodInfo

router = APIRouter()
_PYDANTIC_CONFIG = ConfigDict(extra="forbid", populate_by_name=True)


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class ModelsResponse:
    """Available generation methods payload."""

    methods: list[MethodInfo]
    default: str | None


_MODELS_RESPONSE = TypeAdapter(ModelsResponse)


@router.get("/models")
def list_models() -> JSONResponse:
    """List available head generation methods."""
    payload = ModelsResponse(methods=registry.list_available(), default=registry.default_method_id)
    return JSONResponse(_MODELS_RESPONSE.dump_python(payload, mode="json", by_alias=True))
