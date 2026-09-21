"""Standard-library logging and per-run local JSON persistence."""
import asyncio
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import tempfile


@dataclass
class Context:
    fields: dict = field(default_factory=dict)


class ContextLogger:
    def __init__(self):
        self.log = logging.getLogger('drama')

    def with_context_fields(self, ctx, values):
        ctx.fields.update(values)

    def _emit(self, level, ctx, message):
        self.log.log(level, '[%s] %s', getattr(ctx, 'fields', {}), message)
        if level >= logging.WARNING:
            from .diagnostics import save_event
            save_event('validation_events', level=logging.getLevelName(level),
                       context=dict(getattr(ctx, 'fields', {})), text=message)

    def info_context(self, ctx, message):
        self._emit(logging.INFO, ctx, message)

    def warning_context(self, ctx, message):
        self._emit(logging.WARNING, ctx, message)

    def error_context(self, ctx, message):
        self._emit(logging.ERROR, ctx, message)

    def error(self, message):
        self.log.error(message)


logger = ContextLogger()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, suffix='.tmp', delete=False) as f:
            name = f.name
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


class LocalStore:
    def __init__(self, root=None):
        root = root or os.environ.get('DRAMA_OUTPUT_DIR')
        if not root:
            raise ValueError('Set DRAMA_OUTPUT_DIR or pass a local output directory')
        self.root = Path(root).resolve()

    def path(self, project_id, relative):
        # Keep each run and story isolated; same APID in different runs cannot collide.
        if not project_id or Path(project_id).name != project_id or project_id in ('.', '..'):
            raise ValueError('story_id must be a nonempty single directory name')
        return self.root / 'local_store' / project_id / relative


class LocalFile:
    @classmethod
    async def download(cls, ctx, store, **keys):
        path = store.path(keys['project_id'], cls.relative_path(**keys))
        def read():
            with path.open(encoding='utf-8') as f:
                return cls(**json.load(f))
        return await asyncio.to_thread(read)

    async def upload(self, ctx, store, **keys):
        path = store.path(keys['project_id'], self.relative_path(**keys))
        await asyncio.to_thread(atomic_json, path, asdict(self))


@dataclass
class ScriptContentFile(LocalFile):
    result: str

    @classmethod
    def relative_path(cls, project_id, episode_id):
        return f'script_content/episode_{int(episode_id)}.json'


@dataclass
class ScriptAbsFile(LocalFile):
    episodeAbs: list[str]

    @classmethod
    def relative_path(cls, project_id):
        return 'script_abs.json'


@dataclass
class EpisodeOutlineFile(LocalFile):
    episodeOutline: str

    @classmethod
    def relative_path(cls, project_id, episode_id):
        return f'episode_outline/episode_{int(episode_id)}.json'
