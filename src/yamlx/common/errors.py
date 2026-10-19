class YamlxError(Exception):
    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class ModelNotCachedError(YamlxError):
    def __init__(self, repo_id: str, cache_dir: str) -> None:
        super().__init__(
            f"Model '{repo_id}' is not in the local HuggingFace cache "
            f"at {cache_dir}. Download it with: huggingface-cli download {repo_id}",
            reason="model-not-cached",
        )
        self.repo_id = repo_id
        self.cache_dir = cache_dir


class IncompleteDownloadError(ModelNotCachedError):
    def __init__(self, repo_id: str, cache_dir: str, blob_name: str) -> None:
        YamlxError.__init__(
            self,
            f"Model '{repo_id}' has an unfinished download in the local "
            f"HuggingFace cache at {cache_dir}: {blob_name}. "
            f"Download it again with: huggingface-cli download {repo_id}",
            reason="model-download-incomplete",
        )
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.blob_name = blob_name


class WeightsValidationError(YamlxError):
    def __init__(self, path: str, detail: str) -> None:
        super().__init__(
            f"Weights validation failed for {path}: {detail}",
            reason=f"weights-invalid:{detail}",
        )
        self.path = path
        self.detail = detail


class AudioDecodeError(YamlxError):
    def __init__(self, path: str, detail: str) -> None:
        super().__init__(
            f"Cannot decode audio at {path}: {detail}",
            reason=f"audio-decode:{detail}",
        )
        self.path = path
        self.detail = detail


class StreamingStateError(YamlxError):
    def __init__(self, state: str, action: str) -> None:
        super().__init__(
            f"Cannot {action} while the streaming state is '{state}'",
            reason=f"streaming-state:{state}:{action}",
        )
        self.state = state
        self.action = action
