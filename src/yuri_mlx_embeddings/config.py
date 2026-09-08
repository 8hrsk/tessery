from dataclasses import dataclass

from .descriptors import MODEL_ID, PROFILES, REVISION, ModelProfile
from .errors import ConfigurationError, UnsupportedProfileError


@dataclass(frozen=True)
class LoadOptions:
    model: str = MODEL_ID
    revision: str = REVISION
    dimensions: int = 384
    max_length: int = 512

    def __post_init__(self) -> None:
        profile = self.profile
        if (
            self.revision != profile.revision
            or type(self.dimensions) is not int
            or self.dimensions not in profile.output_dimensions
            or type(self.max_length) is not int
            or self.max_length != profile.max_length
        ):
            raise ConfigurationError()

    @property
    def profile(self) -> ModelProfile:
        if not isinstance(self.model, str) or self.model not in PROFILES:
            raise UnsupportedProfileError()
        return PROFILES[self.model]
