import service.shortanime_by_fiction.data_manager.fiction_manager as _fiction_module
import service.shortanime_by_fiction.data_manager.drama_manager as _drama_module
from service.shortanime_by_fiction.data_manager.fiction_manager import FictionDataManager
from service.shortanime_by_fiction.data_manager.drama_manager import DramaDataManager

COS = None


class DataManager(FictionDataManager, DramaDataManager):
    def __init__(self, ctx):
        self.cos = COS
        self.ctx = ctx
        from service.shortanime_by_fiction.data_models import FictionConfig
        self.cfg = FictionConfig()


__all__ = ["DataManager", "COS"]
