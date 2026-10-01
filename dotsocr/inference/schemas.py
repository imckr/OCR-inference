"""
Pydantic schemas for the OpenAI-compatible inference API.
"""

from typing import List, Optional, Union

from pydantic import BaseModel, Field

from dotsocr.config import DEFAULT_MAX_TOKENS, SERVED_MODEL_NAME


class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]


class ResponseFormat(BaseModel):
    type: str = "text"  # "text", "json_object", or "html"


class ChatCompletionRequest(BaseModel):
    model: str = SERVED_MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = 1
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
