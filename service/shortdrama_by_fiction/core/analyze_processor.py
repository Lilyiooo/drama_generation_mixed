import os
import json
from pyexpat.errors import XML_ERROR_TEXT_DECL

from lib.jinja2.utils import F
from trpc.log import logger

from service.shortdrama_by_fiction.data_model import FictionConfig
from service.shortdrama_by_fiction.core.data_manager import DataManager
from service.shortdrama_by_fiction.tools.prompt_manager import prompt_manager
from service.shortdrama_by_fiction.tools.utils import BaseLlmOperation
from service.shortdrama_by_fiction.tools.nlp_utils import extract_number, extract_episode_number,split_episode_and_content
from service.shortdrama_by_fiction.tools.nlp_utils import fix_scenescript

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache")

def to_dict(obj):
    """
    递归地将一个对象（包括其嵌套的对象）转换为字典。
    """
    # 如果是基本类型或None，直接返回
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    # 如果是字典，递归处理它的每个值
    if isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}
        
    # 如果是列表或元组，递归处理它的每个元素
    if isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
        
    # 如果是具有 __dict__ 属性的对象实例（大多数自定义类的实例都有）
    # vars(obj) 相当于 obj.__dict__
    if hasattr(obj, '__dict__'):
        # 核心逻辑：将对象的属性字典进行递归转换
        return {key: to_dict(value) for key, value in vars(obj).items()}
        
    # 如果是其他无法处理的类型（如集合 set），可以根据需要决定如何处理
    # 这里我们简单地将其转换为字符串
    # 也可以选择抛出异常 raise TypeError(f"Type {type(obj)} not serializable")
    return str(obj)


class IPBaseData(object):
    def __init__(self, fname=""):
        # start\develop\clip\end
        self.fname = fname
        self.outline = {}
        #self.parts = None
        #self.heros = {}
        #self.herolines = None
        #self.world = None
        #self.summary = None
        #self.text = None
        #self.episode_scripts = None
        #self.episode_summary = None
        #self.episode_scripts_str = ""
        self.heros_str = ""
        self.world_str = ""
        self.outline_str_dict = None
        self.outline_str = ""
        self.chapters = []  # 原始章节数据
        self.chapters_summary = []  # 原始章节数据摘要
        #self.chapters_summarymedium = None  # 原始章节数据摘要
        #self.chapters_summarylong = None  # 原始章节数据摘要
        self.text_len = 0
    
    def is_field_available(self, field_name):
        """检查字段是否可用"""
        value = getattr(self, field_name, None)
        if value is None:
            return False
        if isinstance(value, (str)) and value in ["", ""]:
            return False
        if isinstance(value, (list, dict)) and not value:
            return False
        return True
    
    def get_available_fields(self):
        """获取所有可用的字段名列表"""
        available_fields = []
        for field_name in self.__dict__:
            if self.is_field_available(field_name):
                available_fields.append(field_name)
        return available_fields


class ShortdramaGenerateData(IPBaseData):
    def __init__(self, fname):
        super().__init__(fname)
        self.mainline = ""
        self.subline = ""
        self.full_story = ""
        self.episode_plan = ""
        self.episode_scripts = ""
        # json.dumps
        self.outline_plan = None
        self.outline_plan_str = ""
        #self.episode_emotionss = None
        #self.feedback = None
        self.text5w = None
        self.num_episode = 0
    
    def to_short_drama_info(self):
        """返回只包含ShortDramaInfo变量的数据"""
        short_drama_info_fields = {
            'fname': self.fname,
            'outline': self.outline,
            'outline_str': self.outline_str,
            'heros_str': self.heros_str,
            'world_str': self.world_str,
            'mainline': self.mainline,
            'subline': self.subline,
            'outline_plan': self.outline_plan, 
            'outline_plan_str': self.outline_plan_str,
            'num_episode': self.num_episode
            
        }
        #'chapters_summary': self.chapters_summary,
        #'chapters': self.chapters,
        #'full_story': self.full_story,
        #'outline_str': self.outline_str,
        #'text_len': self.text_len,
        #'text5w': self.text5w

        # 过滤掉值为None或默认值的字段
        filtered_data = {}
        for key, value in short_drama_info_fields.items():
            if value is not None:
                # 对于字符串，排除""和空字符串
                if isinstance(value, str) and value in ["", ""]:
                    continue
                # 对于列表和字典，排除空容器
                if isinstance(value, (list, dict)) and not value:
                    continue
                filtered_data[key] = value
        
        return filtered_data




    @classmethod
    def from_short_drama_info(cls, data_dict):
        """从字典恢复IPdata数据结构，处理可能缺失的数据"""
        # 创建新的实例，使用字典中的fname或默认值
        fname = data_dict.get('fname', "")
        instance = cls(fname)
        #instance.outline = to_dict(data_dict.get('outline', None) )
        # 定义字段映射和默认值
        field_mappings = {
            'fname': ('fname', ""),
            'outline': ('outline', None),
            'heros_str': ('heros_str', ""),
            'world_str': ('world_str', ""),
            'outline_str': ('outline_str', ""),
            'mainline': ('mainline', ""),
            'subline': ('subline', ""),
            'outline_plan_str': ('outline_plan_str', ""),
            'outline_plan': ('outline_plan', None),
            'num_episode': ('num_episode', 0)
        }
        # 'chapters': ('chapters', []),
        #    'chapters_summary': ('chapters_summary', []),
        # 移除错误的set_full_story调用，_generate_derived_fields方法会处理完整故事的生成
        # 恢复字段值，处理缺失数据
        print("from shortinfo:", data_dict.keys())
        for dict_key, (field_name, default_value) in field_mappings.items():
            if dict_key == "outline_plan":
                print("outline_plan in inputinfo?",dict_key in data_dict)
            if dict_key in data_dict:
                value = data_dict[dict_key]
                if value is None:
                    setattr(instance, field_name, default_value)
                else:
                    if isinstance(value, str) or isinstance(value, dict):
                        setattr(instance, field_name, value)
                    else:
                        print("to_dict of thsi key:",to_dict(value) )
                        setattr(instance, field_name, to_dict(value))
            else:
                # 如果字典中缺少该字段，使用默认值
                setattr(instance, field_name, default_value)
        
        # 从基础数据生成text5w、full_story、outline_str、text_len字段
        ## 从chapters和chapters_summary恢复part的text和summary
        #instance._generate_derived_fields()
        
        return instance

        

    def _update_from_short_drama_info(self, data_dict):
        """从字典更新当前实例的数据"""
        if not isinstance(data_dict, dict):
            raise ValueError("data_dict must be a dictionary")
        
        # 定义完整的字段映射
        field_mappings = {
            'fname': 'fname',
            'outline': 'outline',
            'outline_str_dict': 'outline_str_dict',
            'heros_str': 'heros_str',
            'world_str': 'world_str',
            'chapters': 'chapters',
            'chapters_summary': 'chapters_summary',
            'mainline': 'mainline',
            'subline': 'subline',
            'full_story': 'full_story',
            'outline_plan': 'outline_plan',
            'outline_plan_str': 'outline_plan_str',
            'outline_str': 'outline_str',
            'text_len': 'text_len',
            'episode_plan': 'episode_plan',
            'episode_scripts': 'episode_scripts',
            'text5w': 'text5w'
        }
        
        # 记录更新的字段数量
        updated_fields = 0
        
        # 更新字段值，只更新存在的字段
        for dict_key, field_name in field_mappings.items():
            if dict_key in data_dict:
                value = data_dict[dict_key]
                # 如果值为None，跳过更新（保持原值）
                if value is not None:
                    # 验证数据类型
                    if self._validate_field_value(field_name, value):
                        setattr(self, field_name, value)
                        updated_fields += 1
        
        # 如果更新了chapters或chapters_summary，重新生成派生字段
        if 'chapters' in data_dict or 'chapters_summary' in data_dict:
            self._generate_derived_fields()
        
        # 如果更新了outline，重新恢复part内容
        if 'outline' in data_dict:
            self._restore_part_content_from_chapters()
        
        return self

    def __validate_field_value(self, field_name, value):
        """验证字段值的类型和格式"""
        # 获取当前字段的预期类型
        expected_types = {
            'fname': (str,),
            'outline': (dict,),
            'outline_str_dict': (dict,),
            'heros_str': (str,),
            'world_str': (str,),
            'chapters': (list,),
            'chapters_summary': (list,),
            'mainline': (str,),
            'subline': (str,),
            'full_story': (str,),
            'outline_plan': (dict, type(None)),
            'outline_plan_str': (str,),
            'outline_str': (str,),
            'text_len': (int,),
            'episode_plan': (str,),
            'episode_scripts': (str,),
            'text5w': (dict, type(None))
        }
        
        # 检查字段是否在预期类型中
        if field_name not in expected_types:
            return True  # 未知字段，允许更新
        
        # 验证类型
        expected_type = expected_types[field_name]
        if not isinstance(value, expected_type):
            # 允许None值
            if value is None:
                return True
            # 尝试类型转换
            try:
                if str in expected_type and not isinstance(value, str):
                    value = str(value)
                elif int in expected_type and not isinstance(value, int):
                    value = int(value)
                else:
                    return False
            except (ValueError, TypeError):
                return False
        
        # 额外的格式验证
        if field_name == 'fname' and not value.strip():
            return False  # fname不能为空
        if field_name == 'text_len' and value < 0:
            return False  # text_len不能为负数
        
        return True


class ShortAnalyze(BaseLlmOperation):
    def __init__(
        self,
        ctx,
        novel_id,
        story_id,
        version="v1",
        prompt_version=None,
        reorder_story=False,
        fix_expand_summary=False,
        auto_episode_num=False,
        add_plot=False,
        target_parts=["start", "develop", "hit", "end"],
        textlimit=50000,
        min_perepisode="3",
        num_episode=60,
    ):
        #["起", "承", "转", "合"],
        super().__init__(
            FictionConfig(), stage="all", name="shortdrama_gen", retry_times=1
        )
        print("novel_id in short analyze class init: ", novel_id)
        self.reverse_part_name_map = {"起":"start", "承":"develop", "转":"hit", "合":"end"}
        self.part_name_map = {"start":"起", "develop":"承", "hit":"转", "end":"合"}
        
        self.ctx = ctx
        self.novel_id = novel_id
        self.userstory_id = story_id
        self.prompt_manager = prompt_manager
        self.data_loader = DataManager(ctx)
        self.cfg = FictionConfig()
        self.season_id = 1
        
        self.version = version
        self.min_perepisode = min_perepisode
        self.default_timeout = 120
        self.shortip_len = 12000
        self.shortpart_len = 4200
        self.partsumm_len = 3000
        self.textlimit = 50000
        self.target_parts = target_parts
        self.prompt_version = prompt_version
        self.reorder_story = reorder_story
        self.fix_expand_summary = fix_expand_summary
        self.auto_episode_num = auto_episode_num
        self.add_plot = add_plot
        # self.heros_str = ""
        # self.world_str = ""

        self.IPdata = ShortdramaGenerateData(novel_id)
        self.IPdata.num_episode = num_episode
        self.IPdata.outline = {ii:{} for ii in self.target_parts} 
        self.IPdata.outline_plan = {ii:"" for ii in self.target_parts} 
        self.IPdata.text5w = {ii:"" for ii in self.target_parts} 
         
        self.prompt_version = prompt_version if prompt_version else {}
        self.IPdata.text_len = 0
        self.text_limit = textlimit

    def from_outlinepart_to_info(self, part_data, save_str="", header=""):
        if isinstance(part_data, dict):
            returns = self._format_dict_items(part_data,header=header)
            save_str += returns 
            #for key, value in part_data.items():
            #    save_str += f"{header}{key}： {value}\n"
        else:
            save_str += f"{header}{part_data}\n\n"
        return save_str


    def from_outline_to_str(self, part_name_map=None, outline_str=None):
        if outline_str and len(outline_str)>10:
            self.IPdata.outline_str = outline_str
            return
        # 生成outline_str
        if self.IPdata.outline:
            if not part_name_map:
                part_name_map = self.part_name_map
            
            # 将outline字典转换为字符串表示, markdown
            
            save_str = ""
            for part_name, part_data in self.IPdata.outline.items():
                # 优先用info数据，否则用outline的数据拼接， info也是outline拼的
                if 'info'in part_data and len(part_data['info'])> 5:
                    save_str += f"{part_name_map[part_name]}\n\n{part_data['info']}" 
                else:
                    tmp_str = self.from_outlinepart_to_info(part_data,save_str)
                    save_str += f"{part_name_map[part_name]}\n\n{tmp_str}" 

            self.IPdata.outline_str = save_str.strip()
        else:
            self.IPdata.outline_str = "" 
        return self.IPdata.outline_str 
        

    async def _generate_derived_fields(self, part_name_map=None):
        """从基础数据生成text5w、full_story、outline_str、text_len字段"""
        # 生成full_story
        print("in restore IPdata:setting full_story")
        await self.set_full_story()
        print("loaded data IPdata")
        if len(self.IPdata.chapters)>3:
            # 如果有章节内容，使用章节内容生成完整故事
            text = "\n".join(self.IPdata.chapters)
            if len(text) < 10000:  # 假设文本限制为10000字符
                self.IPdata.full_story = text
            elif self.IPdata.chapters_summary:
                # 如果文本太长，使用章节摘要
                self.IPdata.full_story = "\n".join(self.IPdata.chapters_summary)
            else:
                self.IPdata.full_story = ""
        else:
            #self.set_full_story()
            self.IPdata.full_story = ""
        
        # 生成outline_str
        self.from_outline_to_str(part_name_map)
        
        # 生成text_len
        if self.IPdata.full_story and self.IPdata.full_story != "":
            self.IPdata.text_len = len(self.IPdata.full_story)
        else:
            self.IPdata.text_len = 0
        
        # 生成text5w
        if self.IPdata.outline and "text" in self.IPdata.outline["start"]:
            self.IPdata.text5w = {}
            for part_name, part_data in self.IPdata.outline.items():
                if isinstance(part_data, dict):
                    # 尝试获取最佳文本版本
                    candidates = [
                        part_data.get("text", ""),
                        #part_data.get("textlong", ""),
                        part_data.get("summary", "")
                    ]
                    
                    best_text = ""
                    for candidate in candidates:
                        if len(candidate) > 0 and len(candidate) < 5000:  # 假设文本限制为5000字符
                            best_text = candidate
                            break
                    
                    if not best_text:
                        # 如果没有合适的文本，使用第一个非空候选
                        for candidate in candidates:
                            if candidate:
                                best_text = candidate
                                break
                    
                    self.IPdata.text5w[part_name] = best_text if best_text else ""
                else:
                    self.IPdata.text5w[part_name] = str(part_data) if part_data else ""
        else:
            self.IPdata.text5w = None
        
        # 从chapters和chapters_summary恢复part的text和summary
        self._restore_part_content_from_chapters()
    
    def _restore_part_content_from_chapters(self):
        """从chapters和chapters_summary恢复part的text和summary"""
        if not self.IPdata.outline:
            return
        # 遍历每个part
        for part_name, part_info in self.IPdata.outline.items():
            if not isinstance(part_info, dict):
                continue
            
            # 获取part的chaptersid
            chaptersid = part_info.get('chaptersid', [])
            if not chaptersid:
                continue

            # 从chapters恢复text
            text_parts = []
            for chapter_id in chaptersid:
                if isinstance(chapter_id, int) and 0 <= chapter_id < len(self.IPdata.chapters):
                    text_parts.append(self.IPdata.chapters[chapter_id])
                elif isinstance(chapter_id, str) and chapter_id.isdigit():
                    idx = int(chapter_id)
                    if 0 <= idx < len(self.IPdata.chapters):
                        text_parts.append(self.IPdata.chapters[idx])
            
            if text_parts:
                part_info['text'] = '\n'.join(text_parts)
            
            # 从chapters_summary恢复summary
            summary_parts = []
            for chapter_id in chaptersid:
                if isinstance(chapter_id, int) and 0 <= chapter_id < len(self.IPdata.chapters_summary):
                    summary_parts.append(self.IPdata.chapters_summary[chapter_id])
                elif isinstance(chapter_id, str) and chapter_id.isdigit():
                    idx = int(chapter_id)
                    if 0 <= idx < len(self.IPdata.chapters_summary):
                        summary_parts.append(self.IPdata.chapters_summary[idx])
            
            if summary_parts:
                part_info['summary'] = '\n'.join(summary_parts)

    def _validate_field_value(self, field_name, value):
        """验证字段值的类型和格式"""
        # 获取当前字段的预期类型
        expected_types = {
            'fname': (str,),
            'outline': (dict,),
            'outline_str_dict': (dict,),
            'heros_str': (str,),
            'world_str': (str,),
            'chapters': (list,),
            'chapters_summary': (list,),
            'mainline': (str,),
            'subline': (str,),
            'full_story': (str,),
            'outline_plan': (dict, type(None)),
            'outline_plan_str': (str,),
            'outline_str': (str,),
            'text_len': (int,),
            'episode_plan': (str,),
            'episode_scripts': (str,),
            'text5w': (dict, type(None)),
            'num_episode': (int, )
        }
        
        # 检查字段是否在预期类型中
        if field_name not in expected_types:
            return True  # 未知字段，允许更新
        
        # 验证类型
        expected_type = expected_types[field_name]
        if not isinstance(value, expected_type):
            # 允许None值
            if value is None:
                return True
            # 尝试类型转换
            try:
                if str in expected_type and not isinstance(value, str):
                    value = str(value)
                elif int in expected_type and not isinstance(value, int):
                    value = int(value)
                else:
                    return False
            except (ValueError, TypeError):
                return False
        
        # 额外的格式验证
        if field_name == 'fname' and not value.strip():
            return False  # fname不能为空
        if field_name == 'text_len' and value < 0:
            return False  # text_len不能为负数
        
        return True

    def set_role_info(self, role_info):
        self.IPdata.heros_str = role_info

    def set_story_outline(self, story_outline_str):
        self.IPdata.outline_str = story_outline_str

    async def set_full_story(self, start_idx=0, end_idx=None, isforce=False):
        print("setting full story")
        print("is NOne", self.IPdata.full_story is None, len(self.IPdata.full_story))
        #if self.IPdata.full_story is None  or len(self.IPdata.full_story) <10 or isforce:
        
        chapter_summaries = await self.data_loader.load_contents(
            self.novel_id, start_idx, end_idx, ctype="summary"
        )
        chapter_contents = await self.data_loader.load_contents(
            self.novel_id, start_idx, end_idx, ctype="content"
        )
        
        self.IPdata.chapters_summary = chapter_summaries

        summary = "\n".join(chapter_summaries)

        self.IPdata.chapters = chapter_contents
        text = "\n".join(chapter_contents)
        if len(text) < self.cfg.text_limit:
            self.IPdata.full_story = text
        else:
            self.IPdata.full_story = summary
        print("in setting full story len:", len(self.IPdata.full_story))
        return self.IPdata.full_story

    def _gen_base_proc(self, task, isregen=False):
        try:
            task_type = task["task_type"]
            novel_id = task["novel_id"]

            prompt, optim = prompt_manager.format_prompt(
                stage="analyze",
                name=task_type,
                variables=task["variables"],
                prmtype="raw",
            )
            # prompt 直接用，optim需要拼接
            response = self._gen_llm_proc(
                prompt, optim, task_type=task_type, novel_id=self.userstory_id,isregen=isregen
            )
            return response
        except Exception as e:
            logger.error_context(
                self.ctx, f"func-generate {novel_id}-{task_type} error: {e}"
            )
        return ""

    def _outline_to_string(self, response):
        return json.dumps(response)


    def _v_to_string(self, input):
        return input if isinstance(input, str) else json.dumps(input)
    # not used
    """
    def _kv_to_string(self, kvdict, isemptyahead=False):
        # for input is dict, string, etc.
        save_string = ""
        header = "  " if isemptyahead else ""
        if isinstance(kvdict, dict):
            for hk1, hv1 in kvdict.items():
                save_string += f"\n{header}{hk1}\n{header}--{hv1}"
        else:
            save_string += f"\n{header}"
            save_string += self._v_to_string(kvdict)
        return save_string
    """
    def _format_dict_items(self, dict_obj, header=""):
        """Format dictionary items as key-value pairs."""
        result = ""
        for key, val in dict_obj.items():
            #result += f"\n{header}{key}： {val}" {header}{key}： {value}\n
            result += f"{header}{key}： {val}\n\n" 
        return result

    def _format_list_items(self, list_obj, header=""):
        """Format list items: either as dict or as simple strings."""
        # Check if list contains dicts
        if list_obj and isinstance(list_obj[0], dict):
            result = ""
            for item in list_obj:
                result += self._format_dict_items(item, header)
            return result
        else:
            # Simple list of strings
            return "\n\n".join([f"{str(i)}" for i in list_obj]) + "\n\n"

    def _listkv_to_string(self, listkv, isemptyahead=True):
        # for input is list of dict, dict, string, etc.
        header = "" if isemptyahead else "    "

        if isinstance(listkv, list):
            return self._format_list_items(listkv, header)
        elif isinstance(listkv, dict):
            return self._format_dict_items(listkv, header)
        else:
            return self._v_to_string(listkv)

    def _instance_to_string(self, input):
        save_str = ""
        if isinstance(input, dict):
            for hk, hv in input.items():
                tmp_str = self._listkv_to_string(hv, True)
                save_str += f"{hk}\n\n{tmp_str}\n\n"
                # for inner kv thing use True, hv: list/dict/str
                
        else:
            save_str += self._listkv_to_string(input)
            # save_str += self._v_to_string(input)

        return save_str

    def get_flatstr_of_outline(self):
        returndict = {}
        for k,v in self.IPdata.outline.items():
            rek = self.part_name_map[k]
            returndict[rek] = v["info"]
        return returndict

    async def generate_story_outline(self, episode_number, start_idx=1, end_idx=None):
        self.IPdata.num_episode = episode_number
        print("gen storyoutline, num_eps",self.IPdata.num_episode)
        # 按顺序执行，确保依赖关系正确
        print(self.IPdata)
        await self.set_full_story(start_idx, end_idx)
        
        # get heros - 依赖set_full_story的结果
        await self.generate_heros(start_idx, end_idx)
        # get mainline - 依赖generate_heros的结果
        await self.generate_mainline(start_idx, end_idx)
        if len(self.IPdata.outline_str) < 10 or self.IPdata.outline_str is None:
            task = {
                "idx": 0,
                "novel_id": self.novel_id,
                "task_type": "get_outline",
                "variables": {
                    "full_story": self.IPdata.full_story,
                    "mainline": self.IPdata.mainline,
                    "episode_number": self.IPdata.num_episode
                },
                "maincall": "generate_story_outline"
            }
            tmp_outline = self._gen_base_proc(task)
            outline_plan = {}
            #outline_plan = (
            #    {} if self.IPdata.outline_plan_str is None else self.IPdata.outline_plan_str
            #)
            if self.IPdata.outline_plan is None:
                self.IPdata.outline_plan = {} 
            print("getting gen_story_outline", tmp_outline)
            for partname, vi in tmp_outline.items():
                # cn to en
                repartname = self.reverse_part_name_map[partname]
                outline_plan[partname] = vi["集数规划"]
                # 分集规划存储
                xplan= vi.pop("集数规划")
                print("story 分集规划:partname", partname, xplan)
                if isinstance(xplan, dict) and "分集内容描述" in xplan and "内容" in xplan["分集内容描述"]: 
                    self.IPdata.outline_plan[repartname] =[f"{kx}：{vx}" for kx,vx in xplan["分集内容描述"]["内容"].items()]
                elif isinstance(xplan, dict) and "分集内容描述" in xplan and isinstance(xplan["分集内容描述"], dict):
                    self.IPdata.outline_plan[repartname] ='\n'.join([f"{kx}：{vx}" for kx,vx in xplan["分集内容描述"].items()])
                elif isinstance(xplan, dict) and "分集内容描述" in xplan and isinstance(xplan["分集内容描述"], list): 
                    if isinstance(xplan["分集内容描述"][0], dict):
                        strsave = []
                        for mm in xplan["分集内容描述"]:
                             strsave.append("\n".join([f"{kx}：{vx}" for kx,vx in mm.items()]))
                        self.IPdata.outline_plan[repartname] = '\n'.join(strsave)
                    elif isinstance(xplan["分集内容描述"][0], str):
                        # str 
                        self.IPdata.outline_plan[repartname] = '\n'.join(xplan["分集内容描述"])
                    else:
                        self.IPdata.outline_plan[repartname] = json.dumps(xplan) 
                else:
                    self.IPdata.outline_plan[repartname] = json.dumps(xplan)
                if repartname not in self.IPdata.outline:
                    self.IPdata.outline[repartname] = {}
                # story outline 存储
                self.IPdata.outline[repartname]["info"] = self._listkv_to_string(vi) 
                print("flat outline-part-info:", self.IPdata.outline[repartname]["info"])
                #self.from_outlinepart_to_info(vi) #self._instance_to_string(vi) 
                print("getting info", partname, self.IPdata.outline[repartname]["info"])
                print("getting info from", vi)
            # get_flatstr_of_outline into 起：xxx 
            self.from_outline_to_str() 
            #self.IPdata.outline_str = self._instance_to_string(
            #    self.get_flatstr_of_outline())
            self.IPdata.outline_plan_str = self._instance_to_string(
                outline_plan)
            self.IPdata.outline_plan = outline_plan
            print("IPdatakeys",self.IPdata.__dict__.keys())
        # consist with block/season scripts style
        return self.IPdata.outline_str, self.IPdata.to_short_drama_info()

    async def regenerate_story_outline(
        self, storyinfo, raw_outline, suggestion, 
        role_info, start_idx=1, end_idx=None
    ):
        await self.restore_geninfo(storyinfo)
        self.IPdata.heros_str = role_info
        # the set thing below should not be runned
        await self.set_full_story(start_idx, end_idx)
        # if regen hero setting new heroes/heros, then this will not kept
        await self.generate_heros(start_idx, end_idx)
        if len(role_info) > 10:
            self.IPdata.heros_str = role_info
        # get mainline
        await self.generate_mainline(start_idx, end_idx)
        task = {
            "idx": 0,
            "novel_id": self.novel_id,
            "task_type": "reget_outline",
            "variables": {
                "full_story": self.IPdata.full_story,
                "mainline": self.IPdata.mainline,
                "outline": raw_outline,
                "suggestion": suggestion,
                "num_eps": self.IPdata.num_episode
            },
            "maincall": "regenerate_story_outline"
        }
        self.IPdata.outline = self._gen_base_proc(task, isregen=True)
        self.IPdata.outline_plan = (
            {} if self.IPdata.outline_plan is None else self.IPdata.outline_plan
        )
        for partname, vi in self.IPdata.outline.items():
            self.IPdata.outline_plan[partname] = vi["集数规划"]
        for partname, vi in self.IPdata.outline.items():
            self.IPdata.outline[partname].pop("集数规划")

        self.IPdata.outline_str = self._instance_to_string(
                self.IPdata.outline)
        # consist with block/season scripts style
        return self.IPdata.outline_str, self.IPdata.to_short_drama_info()

    async def generate_mainline(self, start_idx=1, end_idx=None, maincall=""):
        await self.set_full_story(start_idx, end_idx)
        await self.generate_heros(start_idx, end_idx)
        if self.IPdata.mainline is None or len(self.IPdata.mainline) <10 :
            # print("------>full_story for mainline", type(
            #    self.IPdata.full_story),\
            #    len(self.IPdata.full_story), len(self.IPdata.heros_str))
            # print(self.IPdata.full_story[:10])
            task_mainline = {
                "idx": 0,
                "novel_id": self.novel_id,
                "task_type": "get_mainline",
                "variables": {
                    "full_story": self.IPdata.full_story,
                    "heros_str": self.IPdata.heros_str,
                },
                "maincall": "generate_mainline"+maincall
            }
            result = self._gen_base_proc(task_mainline)
            if isinstance(result, dict) and "主线剧情" in result:
                self.IPdata.mainline = result["主线剧情"]
            if isinstance(result, dict) and "副线剧情" in result:
                self.IPdata.subline = result["副线剧情"]

        return self.IPdata.mainline

    async def generate_heros(self, start_idx=1, end_idx=None, maincall=""):
        await self.set_full_story(start_idx, end_idx)
        # get heros
        if len(self.IPdata.heros_str)<10 :
            task_heros = {
                "idx": 0,
                "novel_id": self.novel_id,
                "task_type": "get_heros",
                "variables": {
                    "full_story": self.IPdata.full_story,
                },
                "maincall": "generate_heros"+maincall
            }
            self.IPdata.heros = self._gen_base_proc(task_heros)
            self.IPdata.heros_str = self._instance_to_string(
                self.IPdata.heros)
        return self.IPdata.heros_str, self.IPdata.to_short_drama_info()

    async def regenerate_heros(self, storyinfo, role_info, 
                               outline, suggestion):
        # self.outline = await self.data_loader.load_story_arcs(novel_id)
        # self.outline_str = json.dumps(self.outline)
        
        await self.restore_geninfo(storyinfo)
        await self.set_full_story(1, None)
        params = {
            "RoleInfo": role_info,
            "Outline": outline,
            "Suggestion": suggestion,
            "ChapterAbstract": self.IPdata.full_story,
        }
        task_heros = {
            "idx": 0,
            "novel_id": self.novel_id,
            "task_type": "refine_role_info",
            "variables": params,
            "maincall": "regenerate_heros"
        }
        
        heros = self._gen_base_proc(task_heros, isregen=True)
        self.IPdata.heros_str = self._instance_to_string(heros)

        return self.IPdata.heros_str, self.IPdata.to_short_drama_info()

    async def generate_world_building(self):
        if self.IPdata.full_story: 
            print("full story len:",len(self.IPdata.full_story))
        if not self.IPdata.full_story:
            await self.set_full_story()

        params = {
            "full_story": self.IPdata.full_story,
        }
        task = {
            "idx": 0,
            "novel_id": self.novel_id,
            "task_type": "world_building",
            "variables": params,
            "maincall": "generate_world_building"
        }
        world = self._gen_base_proc(task)
        self.IPdata.world_str = self._instance_to_string(
            world)
        return self.IPdata.world_str

    def _fetch_part_results(self, maincall=""):
        """Fetch LLM results for each part split."""
        part_res = {pi: "" for pi in self.target_parts}

        for partid, partname in enumerate(self.target_parts):
            task_ippart = {
                "idx": 0,
                "novel_id": self.novel_id,
                "task_type": "get_ippart" + str(partid + 1),
                "variables": {
                    "outline": self.IPdata.outline_str,
                    "full_story": self.IPdata.full_story,
                    "mainline": self.IPdata.mainline,
                },
                "maincall": "_fetch_part_results"+maincall
            }
            # if use epsplan will be better?
            part_res[partname] = self._gen_base_proc(task_ippart)
            # Retry if failed?
        print("done get part llm", partname, part_res[partname]['textid'])
        return part_res

    # not used
    def _process_split_short_mode(self, part_res):
        """Process split results for short text mode (offset=0)."""
        if "textid" not in part_res.get("起", {}):
            for ki, ipparti_res in part_res.items():
                if "text" in ipparti_res:
                    self.IPdata.outline[ki]["text"] = ipparti_res["text"]
                    #self.IPdata.outline[ki]["textshort"] = \
                    #        ipparti_res["text"]
                    #self.IPdata.outline[ki]["textmedium"] = \
                    #        ipparti_res["text"]
                    self.IPdata.outline[ki]["textlong"] = ipparti_res["text"]

    def _process_split_long_mode(self, part_res):
        print("in getting partsplit:", self.IPdata.outline)
        print("getting part_split: res",part_res)
        print("testing part_res:", type(part_res))
        print("testing part_res:", part_res.keys())
        """Process split results for long text mode (offset!=0)."""
        print("ipchapters", self.IPdata.chapters[0][:30]) 
        print("ipchapter summ", self.IPdata.chapters_summary[0])
        print("checkoutline:",self.IPdata.outline.keys())
        if self.target_parts[0] not in self.IPdata.outline:
            self.IPdata.outline = {part: {} for part in self.target_parts}
        for ki, ipparti_res in part_res.items():
            print("part_split_res:", ki, ipparti_res['textid'], ipparti_res['text'][:30])
            try:
                # Store chapter IDs
                
                self.IPdata.outline[ki]["chaptersid"] = ipparti_res.get("textid", [])

                # Populate different text versions
                text_configs = [
                    ("summary", self.IPdata.chapters_summary),
                    # ("textmedium", self.IPdata.chapters_summarymedium),
                    # ("textlong", self.IPdata.chapters_summarylong),
                    ("text", self.IPdata.chapters),
                ]
                for tark, refv in text_configs:
                    self.IPdata.outline[ki][tark] = "".join(
                        [
                            f"第{cid}章：{refv[int(cid)-1]}"
                            for cid in ipparti_res.get("textid", [])
                        ]
                    )
            except Exception as e:
                logger.debug("error processing split for %s: %s", ki, e)
                continue

    def _select_best_text_version(self, outline_part):
        """Select best text version based on length constraint.

        Tries priority order: text > textlong > textmedium > summary.
        Returns first candidate under TEXTLIMIT, otherwise returns summary.
        """
        print("in getting text5w", outline_part.keys())
        print("in getting text5w", outline_part["text"][:10], outline_part["summary"][:10])
        print("in getting text5w", self.textlimit) 
        candidates = [
            outline_part.get("text", ""),
            # outline_part.get("textlong", ""),
            # outline_part.get("textmedium", ""),
            outline_part.get("summary", ""),
        ]
        for candidate in candidates:
            if len(candidate) > 0 and len(candidate) < self.textlimit:
                return candidate
        # Fallback: return summary even if empty
        print("getting summary for part", outline_part.keys())
        return outline_part.get("summary", "")

    async def _select_part_text(self):
        """
        Select appropriate text version for a part based on length.
        test5w for each part, in script_gen, script_polish
        """
        # Early return if already populated
        self.IPdata.text5w = {ki:"" for ki in self.target_parts} if not self.IPdata.text5w else self.IPdata.text5w
        print("before select text5w:",self.IPdata.text5w.keys(), self.IPdata.text5w["start"])

        if self.IPdata.text5w != {} and all(
            len(v) > 0 for _, v in self.IPdata.text5w.items()):
            logger.info("self.IPdata.text5w already set")
            return self.IPdata.text5w
        print("set part text5w befor:",self.target_parts[0],self.IPdata.outline[self.target_parts[0]].keys())
        if "text" not in self.IPdata.outline[self.target_parts[0]]:
            if "summary" not in self.IPdata.outline[self.target_parts[0]]:
                print("no text or summary in part", self.IPdata.ouline[self.target_parts[0]])
                await self.set_full_story()
            else:
                self.IPdata.outline[self.target_parts[0]]["text"] =self.IPdata.outline[self.target_parts[0]]["summary"] 
        #print("set part text5w databefor:",self.IPdata.outline[self.target_parts[0]]['text'][:20]) 
        # Populate text5w for each part
        for ki in self.target_parts:
            if "text" not in self.IPdata.outline[ki]:
                text_configs = [
                    ("summary", self.IPdata.chapters_summary),
                    ("text", self.IPdata.chapters),
                ]
                for tark, refv in text_configs:
                    print("get chapter from cid",self.IPdata.outline[ki]["chaptersid"], len(refv))
                    self.IPdata.outline[ki][tark] = "".join(
                        [
                            f"第{cid}章：{refv[int(cid)-1]}"
                            for cid in self.IPdata.outline[ki]["chaptersid"]
                        ])
            self.IPdata.text5w[ki] = self._select_best_text_version(
                self.IPdata.outline[ki]
            )

        return self.IPdata.text5w

    def _summarize_parts(self, maincall=""):
        """Generate summary for each part using LLM."""
        last_summary = "无"
        print("in summ parts: self.text_limit", self.text_limit)
        for ii in self.target_parts:
            if len(self.IPdata.outline[ii]["text"]) < self.text_limit:
                part_story = self.IPdata.outline[ii]["text"]
            else:
                part_story = self.IPdata.outline[ii]["summary"]
            params = {
                "last_part_summary": last_summary,
                "part_story": part_story,
                "outline": self.IPdata.outline_str,
            }
            task = {
                "idx": 0,
                "novel_id": self.novel_id,
                "task_type": "get_partsummary",
                "variables": params,
                "maincall": "_summarize_parts"+maincall
            }
            self.IPdata.outline[ii]["summary"] = self._gen_base_proc(task)
            last_summary = self.IPdata.outline[ii]["summary"]

    async def ip_part_split(self):
        """Split IP into parts and generate summaries for each part."""
        # Step 1: Determine offset
        # offset = self._get_split_offset()
        # Step 2: Fetch LLM results for each part
        part_res = self._fetch_part_results()
        # Step 3: Process results based on offset mode
        self._process_split_long_mode(part_res)
        # Step 4: Select appropriate text: for part,
        await self._select_part_text()

    async def restore_geninfo(self, datainfo):
        #self.IPdata.update_from_short_drama_info(datainfo)
        if datainfo is None:
            print("!!!!in restore geninfo, datainfo None")
            return
        print("datainfo type:", type(datainfo))
        self.novel_id = datainfo.__dict__["fname"] 
        print(datainfo.__dict__)
        
        # 将PartInfo类型的outline字段转换为字典结构
        datainfo_dict = datainfo.__dict__.copy()
        print("inrestore:", datainfo_dict.keys())
        print("inrestore,",datainfo_dict["outline_plan"])
        print("inrestore outline_plan_str,",datainfo_dict["outline_plan_str"])
        if hasattr(datainfo, 'outline') and datainfo.outline:
            # 将PartInfo对象转换为字典
            outline_dict = {ii:{} for ii in self.target_parts}
            outlineplan_dict=  {ii:"" for ii in self.target_parts} 
            for part_name in self.target_parts:
                if hasattr(datainfo.outline, part_name):
                    part_info = getattr(datainfo.outline, part_name)
                    if part_info:
                        outline_dict[part_name] = {
                            'chaptersid': part_info.chaptersid if hasattr(part_info, 'chaptersid') else [],
                            'info': part_info.info if hasattr(part_info, 'info') else "",
                            'expand_summary_list': part_info.expand_summary_list if hasattr(part_info, 'expand_summary_list') else [], 
                            'refine_summary_list': part_info.refine_summary_list if hasattr(part_info, 'refine_summary_list') else [], 
                            'episodes_ids': part_info.episodes_ids if hasattr(part_info, 'episodes_ids') else [],
                            'text':part_info.text if hasattr(part_info, 'text') else "",
                            'summary':part_info.summary if hasattr(part_info, 'summary') else "",
                            'expand_summary':part_info.expand_summary if hasattr(part_info, 'expand_summary') else "",
                            'refine_summary':part_info.refine_summary if hasattr(part_info, 'refine_summary') else "", 
                            'episodes_script':part_info.episodes_script if hasattr(part_info, 'episodes_script') else {},
                            'episodes_script_fix1':part_info.episodes_script_fix1 if hasattr(part_info, 'episodes_script_fix1') else {}, 
                            'episodes_script_fix2':part_info.episodes_script_fix2 if hasattr(part_info, 'episodes_script_fix2') else {}, 
                        }
                    if len(outline_dict[part_name]["refine_summary_list"])>0:
                        outline_dict[part_name]["refine_summary"] = "\n".join(outline_dict[part_name]["refine_summary_list"])
                    if len(outline_dict[part_name]["refine_summary_list"])>0:
                        outline_dict[part_name]["refine_summary"] = "\n".join(outline_dict[part_name]["refine_summary_list"])
                if hasattr(datainfo.outline_plan, part_name): 
                    part_info = getattr(datainfo.outline_plan, part_name)
                    outlineplan_dict[part_name] = part_info 
            datainfo_dict['outline'] = outline_dict
            datainfo_dict['outline_plan'] = outlineplan_dict 

        self.IPdata = ShortdramaGenerateData.from_short_drama_info(datainfo_dict)
        print("getting IPdata outline", self.IPdata.outline_plan)
        print("getting IPdata from_short_drama_info",self.IPdata.chapters[:2])
        self.novel_id = self.IPdata.fname
        await self._generate_derived_fields() 
        print("after derived restore:", self.IPdata)
        print("after derived restore outline part:", self.IPdata.outline["start"].keys()) 
        print("after derived restore outline part:", self.IPdata.outline["start"]['summary'])  
        print("after derived part expand:", self.IPdata.outline["start"]['expand_summary_list']) 
        await self.set_full_story()
        # 不需要调用set_full_story和_generate_derived_fields
        # 因为from_short_drama_info已经处理了数据恢复
        # print("_____update:", self.novel_id, self.IPdata.outline_str)
        # print("____ full_story", len(self.IPdata.full_story))

    async def restore_environment(self, story_outline, 
                                  role_info, world_building):
        self.IPdata.outline_str = story_outline
        # self.IPdata.mainline = story_outline["mainline"]
        self.IPdata.heros_str = role_info
        self.IPdata.world_str = world_building
        await self.set_full_story(0, -1, isforce=False)

    def restore_plan(self, all_plans):
        for ith, partname in enumerate(self.target_parts):
            self.IPdata.outline[partname] = all_plans[partname]

    def makepb_seasoneps_outline_data(self, target="refine"):
        sid = 1
        one_seasoneps_outline = {"season_id": sid, "episodes": []}
        summ_epid = 0
        for partname in self.target_parts:
            idss = self.IPdata.outline[partname]["chaptersid"] # episode_ids
            epss = self.IPdata.outline[partname][f"{target}_summary_list"]
            if len(idss) != len(epss):
                idss = [ii + summ_epid + 1 for ii in range(len(epss))]
            for idsi, epsi in zip(idss, epss):
                one_eps = {
                    "season_id": sid,
                    "episode_id": int(idsi) if isinstance(idsi, str) else idsi,
                    "content": epsi,
                }
                one_seasoneps_outline["episodes"].append(one_eps)
            summ_epid += len(epss)
        return one_seasoneps_outline

    def makepb_eps_outline_data(self, target="refine"):
        sid = 1
        eps_outline = []
        summ_epid = 0
        if len(self.IPdata.outline[self.target_parts[0]]["chaptersid"]) ==0: 
            print("in saving epsoutline, chaptersid is none!!!")
            avg_len = len(self.IPdata.chapters)//4
            for ith, part in enumerate(self.target_parts):
                self.IPdata.outline[part]["chaptersid"] = [str(i) for i in range(ith*avg_len, (ith+1)*avg_len)]

        for partname in self.target_parts:
            print(f"in makepb_eps_outline, with {target} res:", self.IPdata.outline[partname].keys())
            cidss = self.IPdata.outline[partname]["chaptersid"]
            idss = self.IPdata.outline[partname]["episodes_ids"] 
            epss = self.IPdata.outline[partname][f"{target}_summary_list"]
            idss = [ii + summ_epid for ii in range(len(epss))]
            for idsi, epsi in zip(idss, epss):
                one_eps = {
                    "season_id": sid,
                    "episode_id": idsi,
                    "content": epsi,
                    "chapter_from": int(cidss[0]) if len(cidss)>1 else 0,
                    "chapter_to": int(cidss[-1]) if len(cidss)>1 else 0
                }
                eps_outline.append(one_eps)
            summ_epid += len(epss)
        return eps_outline

    def makepb_eps_drama_data(self, target=""):
        sid = 1
        eps_outline = []
        summ_epid = 0
        for partname in self.target_parts:
            if f"episodes_script{target}" not in self.IPdata.outline[partname] or not(isinstance(self.IPdata.outline[partname][f"episodes_script{target}"],dict)):
                eps_outline.append(one_eps = {
                    "season_id": sid,
                    "episode_id": summ_epid,
                    "script_content": "",
                })
                continue 
            epss = self.IPdata.outline[partname][f"episodes_script{target}"]
            #idss = [ii + summ_epid for ii in range(len(epss))]
            #for idsi, epsi in zip(idss, epss):
            for idsi, epsi in self.IPdata.outline[partname][f"episodes_script{target}"].items(): 
                epsi_str = epsi 

                if isinstance(epsi, dict):
                    epsi_list = []
                    for epsid,actions in epsi.items():
                        if isinstance(actions, dict):
                            action_str = ""
                            for actid, action in actions.items():
                                action_str += f"{actid}\n{action}\n"
                            epsi_list.append(f"{summ_epid+1}-{epsid}\n{action_str}")
                        else:
                            epsi_list.append(f"{summ_epid+1}-{epsid}\n{actions}")
                    epsi_str = "\n\n".join(epsi_list)
                elif not isinstance(epsi, str):
                    epsi_str = json.dumps(epsi)
                    
                one_eps = {
                    "season_id": sid,
                    "episode_id": summ_epid,
                    "script_content": epsi_str,
                }

                eps_outline.append(one_eps)
                summ_epid+=1
            #summ_epid += len(epss)
        return eps_outline

    async def generate_episode_outlines(
        self,
        ipdata_dict,
        num_episode,
        role_info,
        world_building,
    ):
        await self.restore_geninfo(ipdata_dict)
        ## 经常模型出错，避免重复跑。如果存在refine或exoand summary，则直接返回
        expand_all_exists,refine_all_exists = True,True
        for partname in self.target_parts:
            if "expand_summary_list" not in self.IPdata.outline[partname] or len(self.IPdata.outline[partname]["expand_summary_list"]) == 0:
                expand_all_exists = False
            if "refine_summary_list" not in self.IPdata.outline[partname] or len(self.IPdata.outline[partname]["refine_summary_list"]) == 0: 
                refine_all_exists = False
            if (not expand_all_exists) and (not refine_all_exists): 
                break
        logger.debug("is already exists eps outlines? refine %s ,expand %s", refine_all_exists, expand_all_exists)
        logger.debug("num eps target %s, save stock %s", num_episode, self.IPdata.num_episode)
        if refine_all_exists and num_episode == self.IPdata.num_episode: 
            print("exist refine episode outline, return")
            return self.makepb_eps_outline_data("refine"), self.IPdata.to_short_drama_info()
        elif expand_all_exists and num_episode == self.IPdata.num_episode: 
            print("exist expand episode outline, return")
            return self.makepb_eps_outline_data("expand"), self.IPdata.to_short_drama_info()

        ## 如果不存在剧本则xxx
        print("before gen epis plan: outline_str", self.IPdata.outline_str)
        self.IPdata.num_episode = num_episode
        print("gen episodeoutline, num_eps",self.IPdata.num_episode)
        self.IPdata.heros_str = role_info if len(role_info) > 10 else self.IPdata.heros_str 
        self.IPdata.world_str = world_building if len(world_building) > 10 else self.IPdata.world_str 
        print("generating eps outline:", len(self.IPdata.heros_str), len(self.IPdata.world_str))
        all_episodeplan_list = []
        try:
            await self.ip_part_split()
        except Exception as e:
            logger.error_context(self.ctx, f"Error part split: {e}, using defulat")
            self.IPdata.outline[self.target_parts[0]] = {
                    "chaptersid":[i for i in range(len(self.IPdata.chapters))],
                    "text": self.IPdata.full_story,
                    "summry": self.IPdata.full_story,
                    "info": self.IPdata.outline_str,
                    }
        print("finish part split")
        try:
            self.ip_part_expand()
            print("finish part expand")
            offset_id = 0
            
            for partname, vi in self.IPdata.outline.items():
                print("after expand",vi.keys())
                # only process if expand_exists and is non-empty
                expand_list = (
                    vi.get("expand_summary_list",[])
                    if isinstance(vi, dict) else None
                )
                print(f"{partname} expand result:", expand_list)
                if not expand_list:
                    print(f"{partname} expand not valid, use raw:", vi.keys())
                    if "outline_plan" in vi and partname in vi["outline_plan"]: 
                        expand_list =  vi["outline_plan"][partname]
                        print(f"{partname}, expand raw :{vi['outline_plan'][partname]}")
                    continue
                part_plan_add = [
                    f"第{ith + offset_id}集 {epi}"
                    for ith, epi in enumerate(expand_list)
                ]
                # use 
                #part_plan_add = expand_list
                self.IPdata.outline[partname]["episodes_ids_gen"] = [
                    f"{extract_number(epi)+offset_id}"
                    for ith, epi in enumerate(expand_list)
                ]
                self.IPdata.outline[partname]["episodes_ids"] = [
                    f"{ith+1+offset_id}"
                    for ith, epi in enumerate(expand_list)
                ]
                if self.IPdata.outline[partname]["episodes_ids_gen"]!= self.IPdata.outline[partname]["episodes_ids"]: 
                    print(f"getting {partname} epsids error: {self.IPdata.outline[partname]['episodes_ids']}, \
                          {self.IPdata.outline[partname]['episodes_ids_gen']}")
                
                all_episodeplan_list.extend(part_plan_add)
                offset_id += len(expand_list)

        except Exception as e:
            logger.error_context(self.ctx, f"part expand Error: {e}")
            """
            if len(all_episodeplan_list) <1:
                print("getting no plan, use raw")
                for partname, vi in self.IPdata.outline.items():
                    if "outline_plan" in vi and partname in vi["outline_plan"]: 
                        expand_list =  vi["outline_plan"][partname]
                        print(f"{partname}, expand raw :{vi['outline_plan'][partname]}")
                        all_episodeplan_list.extend(expand_list)
                        print("raw plan", partname, expand_list)
            """
            return {}, self.IPdata.to_short_drama_info()

        episode_plan = "\n".join(all_episodeplan_list)
        print("getting episode plan", episode_plan)
        print("after part expand, get eps plan", episode_plan)
        
        """
        try:
            _ = self.refine_episode_outline(episode_plan)
        except Exception as e:
            logger.error_context(self.ctx, f"Error: {e}")
            expand_oneseason = self.makepb_eps_outline_data("expand")
            # print("INNNNNNN expand", expand_oneseason)
            return expand_oneseason, self.IPdata.to_short_drama_info()

        expand_oneseason = self.makepb_eps_outline_data("refine")
        # print("INNNNNNN refine", expand_oneseason)
        return expand_oneseason, self.IPdata.to_short_drama_info()
        """
        expand_oneseason = self.makepb_eps_outline_data("expand")
        return expand_oneseason, self.IPdata.to_short_drama_info()

    def refine_episode_outline(self, outline, maincall=""):
        all_episodeplan_list = []
        part_epss = []
        print("refine all eps plan",self.IPdata.outline[self.target_parts[0]].keys())
        expand_plan = []
        for partname,vi in self.IPdata.outline.items():
            print("in refinesumm,expandsumm",partname,vi["expand_summary_list"])
            expand_plan.extend(vi["expand_summary_list"])
        try:
            variables = {
                "eps_plan": self.IPdata.num_episode,
                "min_pereps": self.min_perepisode,
                "world_building": self.IPdata.world_str,
                "heros_str": self.IPdata.heros_str,
                "full_story": self.IPdata.full_story,
                "raw_plan": outline,
                "outline": self.IPdata.outline_str,
            }

            task = {
                "idx": 0,
                "novel_id": self.novel_id,
                "task_type": "refine_eps_plan",
                "variables": variables,
                "maincall": "refine_episode_outline"+maincall
            }
            res = self._gen_base_proc(task)
            offset_id = 1
            print("refine eps plan:", res.keys())
            print("refine eps plan all", res) 
            print("raw plan:", outline)
            print(self.IPdata.outline["start"].keys())
            print("refine output", res)
            print("outlinestr",self.IPdata.outline_str)
            # todo fix part 提纲
            for partname in self.target_parts:
                cnpartname = self.reverse_part_name_map[partname]
                part_expand_summary_list = res[f"{cnpartname}提纲"]
                # normalize to consistent format (list of strings)
                epsid_list, part_expand_summary_list = self._parse_expand_summary_list(
                    part_expand_summary_list
                )
                print("refinsumm ", partname, part_expand_summary_list)
                self.IPdata.outline[partname][
                    "refine_summary_list"
                ] = part_expand_summary_list

                part_expand_summary = "\n".join(part_expand_summary_list)
                self.IPdata.outline[partname]["refine_summary"] = \
                    part_expand_summary

                part_plan_add = [
                    f"第{ith + offset_id}集\n {epi}"
                    for ith, epi in enumerate(part_expand_summary_list)
                ]
                all_episodeplan_list.extend(part_plan_add)
                offset_id += len(part_expand_summary_list)
                part_epss.append("\n".join(part_expand_summary_list))
                print(f"for {partname} refinesumm:{self.IPdata.outline[partname]['refine_summary_list']}")
        except Exception as e:
            logger.error_context(
                self.ctx, f"refine epi outline task Error: {e}\n{outline}"
            )
            return []

        return all_episodeplan_list

    async def regenerate_episode_outline(
        self,
        ipdata_dict,
        role_info,
        world_building,
        eps_outlines,
        suggestion
        # num_episode,role_info,world_building,
    ):
        # self.heros_str = role_info
        await self.restore_geninfo(ipdata_dict)
        all_episodeplan_list = []
        try:
            variables = {
                "world_building": world_building,
                "heros_str": role_info,
                "full_story": self.IPdata.full_story,
                "raw_plan": eps_outlines,
                "outline": self.IPdata.outline_str,
                "Suggestion": suggestion,
            }
            #
            task = {
                "idx": 0,
                "novel_id": self.novel_id,
                "task_type": "refine_eps_plan_bysugession",
                "variables": variables,
                "maincall": "regenerate_episode_outline"
            }
            res = self._gen_base_proc(task, isregen=True)
            offset_id = 1
            for partname in self.target_parts:
                repartname = self.part_name_map[partname]
                part_expand_summary_list = res[f"{repartname}提纲"]
                # normalize to consistent format (list of strings)
                epsid_list, part_expand_summary_list = self._parse_expand_summary_list(
                    part_expand_summary_list
                )
                self.IPdata.outline[partname][
                    "refine_summary_list"
                ] = part_expand_summary_list

                part_expand_summary = "\n".join(part_expand_summary_list)
                self.IPdata.outline[partname]["refine_summary"] = \
                    part_expand_summary

                part_plan_add = [
                    f"第{ith + offset_id}集\n {epi}"
                    for ith, epi in enumerate(part_expand_summary_list)
                ]
                self.IPdata.outline[partname]["episodes_ids_gen"] = epsid_list 
                #self.IPdata.outline[partname]["episodes_ids"] = [
                #    f"{extract_number(epi)+offset_id}"
                #    for ith, epi in enumerate(part_expand_summary_list)
                #]
                all_episodeplan_list.extend(part_plan_add)
                offset_id += len(part_expand_summary_list)

        except Exception as e:
            logger.error_context(
                self.ctx, f"refine epi outline task Error: {e} \
                    {eps_outlines}"
            )

        expand_oneseason = self.makepb_eps_outline_data("refine")
        return expand_oneseason, self.IPdata.to_short_drama_info()

    async def regenerate_episode_outline_single(
        self,
        role_info,
        world_building,
        tareps_id,
        tareps_outline,
        suggestion
        # num_episode,role_info,world_building,
    ):
        # self.heros_str = role_info
        all_episodeplan_list = []

        total_refine_summ = []
        for partname in self.target_parts:
            total_refine_summ.append(self.IPdata.outline[partname
                                     ]["refine_summary"])
        refine_summ = "\n".join(total_refine_summ)
        try:
            variables = {
                "tareps_id": tareps_id,
                "world_building": world_building,
                "heros_str": role_info,
                "full_story": self.IPdata.full_story,
                "raw_plan": refine_summ,
                "outline": self.IPdata.outline_str,
                "Suggestion": suggestion,
                "raw_target_plan": tareps_outline,
            }
            #
            task = {
                "idx": 0,
                "novel_id": self.novel_id,
                "task_type": "refine_oneeps_plan_bysugession",
                "variables": variables,
                "maincall": "regenerate_episode_outline_single"
            }
            res = self._gen_base_proc(task, isregen=True)["提纲"]
            offset_id = 1
            all_episodeplan_list = []
            for partname in self.target_parts:
                if str(tareps_id) in self.IPdata.outline[partname
                        ]["episodes_ids"]:
                    index = self.IPdata.outline[partname]["episodes_ids"
                            ].index(str(tareps_id))
                    self.IPdata.outline[partname]["refine_summary_list"
                            ][index] = res
                    part_expand_summary = "\n".join(res)
                    self.IPdata.outline[partname][
                        "refine_summary"
                    ] = part_expand_summary

                # normalize to consistent format (list of strings)
                part_expand_summary_list = self.IPdata.outline[partname][
                    "refine_summary_list"
                ]
                part_plan_add = [
                    f"第{ith + offset_id}集\n {epi}"
                    for ith, epi in enumerate(part_expand_summary_list)
                ]
                all_episodeplan_list.extend(part_plan_add)
                offset_id += len(part_expand_summary_list)

        except Exception as e:
            logger.error_context(
                self.ctx, f"refine epi outline task Error: {e}\n{refine_summ}"
            )
        expand_oneseason = self.makepb_eps_outline_data("refine")

        return expand_oneseason, self.IPdata

    def _determine_expand_prmkey(self, partname, defaultkey, updateinfo):
        """Determine the prompt key (prmkey) for part expansion.

        Logic:
        1. If updateinfo is provided, use 'update' branch
        2. If short IP + short part OR add_plot setting, use 'short' branch
        3. Otherwise use default

        Returns: str, the prompt key to use for LLM call
        """
        # Step 1: check if update mode
        if updateinfo is not None:
            prmkey = f"{defaultkey}_update"
        else:
            prmkey = defaultkey

        # Step 2: check if short IP + short part text
        part_text_len = len(self.IPdata.outline.get(partname, 
                            {}).get("text", ""))
        is_short_ip_and_part = (
            self.IPdata.text_len < self.shortip_len
            and part_text_len < self.shortpart_len
        )

        # Step 3: apply shortIP branch if conditions met
        if (is_short_ip_and_part or self.add_plot) and prmkey == defaultkey:
            prmkey = f"{defaultkey}_short"
            logger.info(
                f"Using short variant for part {partname}: \
                prmkey={prmkey}"
            )

        return prmkey

    def _flatten_string(self, obj):
        """Flatten any object to string: list -> join, else -> str()."""
        if isinstance(obj, list):
            return "".join(obj)
        return str(obj) if obj else ""

    def _extract_deepseek_format(self, epsi):
        """Extract episode info from DeepSeek format: '集数' and '内容' keys."""
        eps_num = str(epsi.get("集数", ""))
        eps_content = epsi.get("内容", "")
        return eps_num, eps_content

    def _extract_other_format(self, epsi):
        """Extract episode info from other LLM formats (key-value pairs)."""
        eps_num = None
        eps_content = None

        # Try to find episode number in keys
        for key, val in epsi.items():
            if len(epsi.keys()) == 1:
                # Single key-value: extract episode number from key
                eps_num = str(extract_number(key))
                eps_content = val
                break
            elif any(kw in key.lower() for kw in ["第", "集", "episode"]):
                # Key contains episode indicator
                eps_num = str(extract_number(key))
                eps_content = val
                break

        return eps_num, eps_content

    def _normalize_episode_item(self, epsi):
        """Normalize single episode dict into format string '第N集\\n内容'."""
        try:
            print("expand parse:", epsi)
            # Dispatch based on format
            if "集数" in epsi:
                eps_num, eps_content = self._extract_deepseek_format(epsi)
            else:
                eps_num, eps_content = self._extract_other_format(epsi)

            # Flatten content (handle list inputs)
            eps_content = self._flatten_string(eps_content)
            print("normalized episode:", eps_num, eps_content)
            # Validate and return
            if eps_num and eps_content:
                #return f"第{eps_num}集\n{eps_content}"
                return eps_num, eps_content
            else:
                logger.debug(
                    "skip malformed expand item (missing num or content): \
                    %s",
                    epsi,
                )
                return None
        except Exception as e:
            logger.debug("error normalizing expand item %s: %s", epsi, e)
            return None

    def _parse_expand_summary_list(self, summ_list):
        """Parse and normalize multiple LLM response into consistent format.

        Handles:
        - list[str]: direct list of strings (flatten nested lists)
        - list[dict]: each dict may have '集数'/'内容' or '第N集'/'content'
        - dict: treat as single item
        Returns: list[str] in format "第N集\\n内容"
        """
        normalized,epsid_list = [],[]
        # Early exit for empty input
        if not summ_list or (isinstance(summ_list, list) and
                len(summ_list) == 0):
            return [],[]

        # Detect format: dict-based vs string-based
        is_dict_format = False
        try:
            is_dict_format = isinstance(summ_list, dict) or (
                isinstance(summ_list, list)
                and len(summ_list) > 0
                and isinstance(summ_list[0], dict)
            )
        except (IndexError, TypeError):
            is_dict_format = False

        # Process string-based format (simple list flattening)
        if not is_dict_format:
            print("in parse expand not a dict:", summ_list)

            for item in summ_list:
                itemstr = self._flatten_string(item)
                expid, content = split_episode_and_content(itemstr)
                print("in parse expandsumm:", item)
                print("in parse expandsumm flatten:", itemstr)
                if content:
                    normalized.append(content)
                #expid = extract_episode_number(item)
                
                epsid_list.append(expid) 
            print("in parse expand epsid:", summ_list)
            #epsid_list = [str(i) for i in range(len(normalized))]
            return epsid_list, normalized

        # Process dict-based format (LLM response with episode info)
        items = summ_list if isinstance(summ_list, list) else [summ_list]
        for epsi in items:
            if isinstance(epsi, dict):
                epsid, result = self._normalize_episode_item(epsi)
                if result:
                    normalized.append(result)
                    epsid_list.append(epsid)

        return epsid_list, normalized

    def ip_part_expand(self,maincall=""):
        prev_expand_summary = "无"
        if self.IPdata.outline:
            print("inpart expand", self.IPdata.outline.keys(),self.IPdata.outline["start"].keys())
        offset_id = 0

        if "起" in self.IPdata.outline_plan:
            # remap dict:
            for cnpart, enpart in self.reverse_part_name_map.items():
                if cnpart in self.IPdata.outline_plan:
                    self.IPdata.outline_plan[enpart] = self.IPdata.outline_plan.pop(cnpart)
        
        if len(self.IPdata.outline_plan[self.target_parts[0]]) < 5 and len(self.IPdata.outline_plan_str)>20:
            now_str = self.IPdata.outline_plan_str 
            for ith, partname in enumerate(self.target_parts):
                repartname = self.part_name_map[partname]
                if ith ==0:
                    continue
                part_before, part_left = now_str.split(f"{repartname}\n\n")
                logger.debug(f"splitting outline_str for outline {repartname} {part_before}")
                before_partname = self.part_name_map[self.target_parts[ith-1]]
                now_str= part_left
                self.IPdata.outline_plan[before_partname] = {"总集数建议":"", "分集内容描述":part_before}
            now_partname = self.part_name_map[self.target_parts[-1]]
            self.IPdata.outline_plan[now_partname] = {"总集数建议":"", "分集内容描述":now_str}
            logger.debug(f"splitting outline_str for outlineall getting: {self.IPdata.outline_plan}")
                    
        # partnameflat getting
        flatpartplan_map = {}
        not_valid = False
        for ith, partname in enumerate(self.target_parts):
            repartname = self.part_name_map[partname]
            if repartname not in self.IPdata.outline_plan:
                print("outlinekey:",self.IPdata.outline_plan.keys())
                not_valid = True
                break
            if isinstance(self.IPdata.outline_plan[repartname], str):
                flatpartplan_map[partname] = self.IPdata.outline_plan[repartname]
            else:
                flatpartplan_map[partname] = json.dumps(self.IPdata.outline_plan[repartname])
        logger.debug(f"getting flatpartplan1: {flatpartplan_map}")
        # if not valid:
        if not_valid:
            num_list = []
            print("usding force episdoe split:",self.IPdata.num_episode,", 2/5,3/5,7/20,3/20")
            num_list.append(int(self.IPdata.num_episode * 2/5))
            num_list.append(int(self.IPdata.num_episode * 3/5))
            num_list.append(int(self.IPdata.num_episode * 7/20))
            num_list.append(self.IPdata.num_episode- num_list[0]-num_list[1],-num_list[2])
            for num_cnt, partname in zip(num_list, self.target_parts):
                flatpartplan_map[partname] = f"共 {num_cnt} 集"
        logger.debug(f"getting flatpartplan2: {flatpartplan_map}")
        if not self.IPdata.outline_plan or self.IPdata.outline_plan == {} or self.IPdata.outline_plan[self.part_name_map[self.target_parts[0]]] == {}:
            self.IPdata.outline_plan = flatpartplan_map


        for partid, partname in enumerate(self.target_parts):  
            vi =  self.IPdata.outline[partname] 
            # use all but not good
            #part_plan = self.IPdata.outline_plan_str if len(self.IPdata.outline_plan[now_partname]) < 5 else self.IPdata.outline_plan[now_partname]
            if isinstance(self.IPdata.outline_plan, dict) and "起" in self.IPdata.outline_plan and self.IPdata.outline_plan["起"] == {}:
                logger.debug(f"using outline_plan empty {now_partname}, {self.IPdata.outline_plan['起']}, using total plan：{self.IPdata.outline_plan_str}")
                print("in part expand self.IPdata.outline_plan", self.IPdata.outline_plan)
            
            logger.debug(f"using outline_plan  {flatpartplan_map[partname]}")
            variables = {
                "partname": partname, 
                "num_episode":self.IPdata.num_episode, 
                "min_perepisode":self.min_perepisode,
                "world_building": self.IPdata.world_str,
                "heros": self.IPdata.heros_str,
                "last_part_episode_plan":prev_expand_summary, 
                "part_summary": vi["summary"],
                "part_plan":flatpartplan_map[partname],
                } #vi["集数规划"] if "集数规划" not in vi else ""
            task = {
                "idx": 0,
                "novel_id": self.novel_id,
                "task_type": f"gen_partexpand_episodeplan{partid+1}",
                "variables": variables,
                "maincall": "ip_part_expand"+maincall
            }
            returnobj = self._gen_base_proc(task)
            logger.debug(f"expand eps outline: {partname}, {returnobj}")
            if "提纲" not in returnobj:
                print("try again, expand eps outline:",partname)
                returnobj = self._gen_base_proc(task)
            part_expand_summary_list = returnobj["提纲"] 
            # normalize to consistent format (list of strings)
            epsid_list, part_expand_summary_list = self._parse_expand_summary_list(
                part_expand_summary_list
            )
            self.IPdata.outline[partname][
                "expand_summary_list"
            ] = part_expand_summary_list
            print("expand summlist:", partname, self.IPdata.outline[partname][
                "expand_summary_list"])
            # all in str format: to consist with deepseek hunyuan model
            self.IPdata.outline[partname]["episodes_ids"] = epsid_list if partid==0 \
                else [str(idx+offset_id) for idx in range(len(part_expand_summary_list))]
            offset_id += len(part_expand_summary_list)
            self.IPdata.outline[partname]["expand_summary"] = \
                "\n".join(part_expand_summary_list)
            prev_expand_summary = self.IPdata.outline[partname]["expand_summary"]
        # finish

    def _extract_script_from_response(self, response_dict, fallback):
        """Extract script dict from various LLM response formats."""
        try:
            if isinstance(response_dict, dict) and "script" in response_dict:
                return response_dict["script"]
        except Exception:
            pass

        # If response is already a dict-like script, use it
        if isinstance(response_dict, dict) and len(response_dict) > 1:
            return response_dict

        # Try to extract the first non-metadata value
        if isinstance(response_dict, dict):
            for k, v in response_dict.items():
                if isinstance(v, dict):
                    return v

        # If error, use fallback
        logger.debug("failed to get script from response, using fallback")
        return fallback

    def determine_ifscript_valid(self,partname, tarkey="episodes_script"):
        ifexists= True
        if "episodes_script" not in self.IPdata.outline[partname] or self.IPdata.outline[partname]["episodes_script"] == {} or self.IPdata.outline[partname]["episodes_script"] == []:
            ifexists = False 
        elif ("episodes_script" in self.IPdata.outline[partname]) and isinstance(self.IPdata.outline[partname]["episodes_script"],dict):
            eps_script_keys = self.IPdata.outline[partname]["episodes_script"].keys
            if "1" not in self.IPdata.outline[partname]["episodes_script"][list(eps_script_keys)[0]] or len(self.IPdata.outline[partname]["episodes_script"][list(eps_script_keys)[0]]["1"])<30:
                ifexists = False
        elif ("episodes_script" in self.IPdata.outline[partname]) and isinstance(self.IPdata.outline[partname]["episodes_script"],list):
            if len(self.IPdata.outline[partname]["episodes_script"][0]) < 30:
                    ifexists = False
        return ifexists

    async def ip_part_script_gen(self, datainfo, role_info, world_building):
        eps_offset = 0
        print("download info", datainfo)
        print("partscript input role", role_info)
        print("partscript input world", world_building)
        await self.restore_geninfo(datainfo)
        ## 经常模型出错，避免重复跑。如果存在script，则直接返回
        fix_all_exists,raw_all_exists = True,True
        for partname in self.target_parts:
            raw_all_exists = self.determine_ifscript_valid(partname,"episodes_script") 
            fix_all_exists = self.determine_ifscript_valid(partname,"episodes_script_fix2") 
            """
            if "episodes_script" not in self.IPdata.outline[partname] or self.IPdata.outline[partname]["episodes_script"] == {} or self.IPdata.outline[partname]["episodes_script"] == []:
                raw_all_exists = False 
            elif ("episodes_script" in self.IPdata.outline[partname]) and isinstance(self.IPdata.outline[partname]["episodes_script"],dict):
                eps_script_keys = self.IPdata.outline[partname]["episodes_script"].keys
                if "1" not in self.IPdata.outline[partname]["episodes_script"][list(eps_script_keys)[0]] or len(self.IPdata.outline[partname]["episodes_script"][list(eps_script_keys)[0]]["1"])<30:
                    raw_all_exists = False
            elif ("episodes_script" in self.IPdata.outline[partname]) and isinstance(self.IPdata.outline[partname]["episodes_script"],list):
                if len(self.IPdata.outline[partname]["episodes_script"][0]) < 30:
                    raw_all_exists = False
            
            if "episodes_script_fix2" not in self.IPdata.outline[partname] or self.IPdata.outline[partname]["episodes_script_fix2"] == {} or self.IPdata.outline[partname]["episodes_script_fix2"] == []: 
                fix_all_exists = False
            """
            if (not raw_all_exists) and (not fix_all_exists): 
                break
        logger.debug("is already exists eps sript? raw %s, fix %s", raw_all_exists, fix_all_exists)
        #logger.debug(self.ctx, f"num eps target {num_episode}, stock {self.IPdata.num_episode}")

        if fix_all_exists : 
            print("exist fix2 episode outline, return")
            return self.makepb_eps_drama_data("_fix2"), self.IPdata.to_short_drama_info()
        elif raw_all_exists: 
            print("exist expand episode outline, return")
            return self.makepb_eps_drama_data(), self.IPdata.to_short_drama_info()


        print("scriptgen after restore IPata outline",self.IPdata.outline)
        print("scriptgen after restore IPdata chapters:",len(self.IPdata.chapters))
        self.IPdata.heros_str = role_info
        self.IPdata.world_str = world_building
        await self._select_part_text()
        print("scriptgen part text",self.IPdata.text5w["start"][:100])
        last_episode_script = ""
        #for partname, vi in self.IPdata.outline.items():
        for partid, partname in enumerate(self.target_parts):
            vi = self.IPdata.outline[partname] 
            #if partname not in self.target_parts:
            #    continue
            text5w = self.IPdata.text5w[partname]
            print("scriptgen vi keys", vi.keys()) 
            episode_core_story = vi["summary"]
            
            episode_plans = episode_plans = "\n".join([f"第{ii}集：{jj}" for ii,jj in zip(vi["episodes_ids"],vi["refine_summary_list"])]) 
            logger.debug("part refine summlist %s ", episode_plans)
            if len(episode_plans)<200:
                # < 4*40集=80 空时
                episode_plans = "\n".join([f"第{ii}集：{jj}" for ii,jj in zip(vi["episodes_ids"],vi["expand_summary_list"])]) 
                logger.debug("part expand summlist %s", episode_plans)
            print("expand plan", vi["expand_summary_list"])
            print("llm input: scriptgen input plan", episode_plans)
            # 5 prm
            print("llm input: part, text5w, corestory:", partname, text5w[:100],episode_core_story[:100])
            print("last type", type(last_episode_script))

            variables = {
                "partname": partname,
                "part_text": text5w,
                "part_summary": episode_core_story,
                "part_plan": episode_plans,
                "partscript_last": last_episode_script
            }

            task = {
                "idx": 0,
                "novel_id": self.novel_id,
                "task_type": f"gen_partscript{partid+1}",
                "variables": variables,
                "maincall": 'ip_part_script_gen'
            }
            print("before genscript", partname, episode_plans)
            part_script_dict = self._gen_base_proc(task)
            print("scriptgen:, part_script_dict", part_script_dict)
            # key:第x集->x
            delkeys = []

            for ki, vi in part_script_dict.items():
                if "第" in ki:
                    newki = ki.replace("第", "").replace("集", "")
                    part_script_dict[newki] = vi
                    delkeys.append(ki)
                
            if len(delkeys) > 0:
                for ki in delkeys:
                    del part_script_dict[ki]
            print("scriptgen:, after delkey", part_script_dict) 
            # 对每集-每场
            self.IPdata.outline[partname]["episodes_script"] = {
                f"{extract_number(ki)+eps_offset}": vi
                for ki, vi in part_script_dict.items()
            }
            print("genepisodes_script",partname, self.IPdata.outline[partname]["episodes_script"])
            self.IPdata.outline[partname]["episodes_ids"] = [
                f"{extract_number(ki)+eps_offset}" for ki in part_script_dict.keys()
            ]
            self.IPdata.outline[partname]["episodes_ids_gen"] = [
                f"{extract_number(ki)}" for ki in part_script_dict.keys()
            ]
            print("script_gen:",partname, self.IPdata.outline[partname]["episodes_ids"])
            print(self.IPdata.outline[partname]["episodes_ids"][-1])

            last_episode_script= self.IPdata.outline[partname]["episodes_script"][
                    self.IPdata.outline[partname]["episodes_ids"][-1]]
            
            eps_offset += len(part_script_dict)

        # try:
        #    self.ip_part_script_polish()
        # except Exception as e:
        #    print(f"polish error {e}")

        return self.makepb_eps_drama_data(), self.IPdata.to_short_drama_info()

    async def ip_part_script_polish(self):
        """Polish scripts for all parts and episodes.
        - init episode_script_fix1/fix2 dicts
        - _polish_one_episode: polish single episode (logic + format)
        - _save_part_polish_result: save and assemble part results
        """
        await self._select_part_text()
        last_episodes_script = None
        last_episode_ids = None
        for partname, vi in self.IPdata.outline.items():
            if partname not in self.target_parts:
                continue
            partid = self.target_parts.index(partname)

            logger.info("=" * 50 + " %s " + "=" * 50, partname)
            # Prepare part-level data
            part_summary = vi.get("expand_summary", "")
            episodes_script = vi.get("episodes_script", {})
            part_text = self.IPdata.text5w[partname]
            # init tracking structures
            self.IPdata.outline[partname][
                "episodes_script_raw"
            ] = episodes_script.copy()
            self.IPdata.outline[partname]["episodes_script_fix1"] = {
                ei: None for ei in vi.get("episodes_ids", [])
            }
            self.IPdata.outline[partname]["episodes_script_fix2"] = {
                ei: None for ei in vi.get("episodes_ids", [])
            }

            episcriptfix_prev = ""
            for idx, ei in enumerate(vi.get("episodes_ids", [])):
                episcript = json.dumps(episodes_script.get(ei, {}))
                # set prev episode:1)非首集用当前部分上集; 2)首集但非1部，上部最后集;3）1部1集
                episcript_prev = ""
                if idx > 0:
                    episcript_prev = episodes_script.get(
                        vi["episodes_ids"][idx - 1], ""
                    )
                elif partid > 0:
                    episcript_prev = last_episodes_script.get(last_episode_ids, "")

                # logic polish
                fix_script = self._polish_logic_one_episode(
                    partname, ei, episcript, episcript_prev, part_summary, part_text
                )
                self.IPdata.outline[partname]["episodes_script_fix1"][ei] = fix_script
                # format polish
                polish_formmat = self._polish_format_one_episode(
                    partname, ei, fix_script, episcriptfix_prev, part_summary, part_text,maincall="ip_part_script_polish"
                )
                self.IPdata.outline[partname]["episodes_script_fix2"][
                    ei
                ] = polish_formmat

                # apply header fix and update outline
                fix_header_formmat = {
                    actid: fix_scenescript(scripti, ei, actid)
                    for actid, scripti in polish_formmat.items()
                }
                self.IPdata.outline[partname]["episodes_script"][
                    ei
                ] = fix_header_formmat
                episcriptfix_prev = fix_header_formmat

            # save and assemble part results
            # self._save_part_polish_result(partname, vi)
            # last_episodes_script = vi["episodes_script"]
            # last_episode_ids = vi["episodes_ids"][-1]
        # finish

    def _polish_logic_one_episode(
        self, partname, ei, episcript, episcript_prev, part_summary, part_text
    ):
        """Polish one episode for logic issues."""
        part_content = part_summary if len(part_text) > 50000 else part_text
        task = {
            "idx": 0,
            "novel_id": self.novel_id,
            "task_type": "gen_polishlogic",
            "variables": {
                "eps_id": ei,
                "eps_script": episcript,
                "partname": partname,
                "part_text": part_content,
                "last_eps_script": episcript_prev,
                "heros_str": self.IPdata.heros_str,
                "full_story": self.IPdata.full_story,
            },
        }
        try:
            polish_logic_raw = self._gen_base_proc(task)
        except Exception:
            logger.exception(
                "BOT call failed for gen_polishlogic in episode %s-%s", partname, ei
            )
            polish_logic_raw = {}

        # normalize response
        fix_script = self._extract_script_from_response(polish_logic_raw, episcript)
        logger.info("[%s 第%s集 fixed logic]", partname, ei)
        return fix_script

    def _polish_format_one_episode(
        self, partname, ei, fix_script, episcriptfix_prev, part_summary, part_text, maincall=""
    ):
        """Polish one episode for format issues."""
        task = {
            "idx": 0,
            "novel_id": self.novel_id,
            "task_type": "gen_polishformat",
            "variables": {
                "eps_id": ei,
                "eps_script": fix_script,
                "partname": partname,
                "part_text": part_text,
                "last_eps_script": episcriptfix_prev,
                "heros_str": self.IPdata.heros_str,
                "full_story": self.IPdata.full_story,
            },
            "maincall": "_polish_format_one_episode"+maincall
        }
        try:
            polish_formmat_raw = self._gen_base_proc(task)
        except Exception:
            logger.exception(
                "call failed for gen_polishformmat in episode %s-%s", partname, ei
            )
            polish_formmat_raw = {}

        # fallback to sumBOT if needed

        polish_formmat = self._extract_script_from_response(
            polish_formmat_raw, fix_script
        )
        logger.info("[第%s集 fixed formmat]", ei)
        return polish_formmat

    def _extract_script_from_response(self, response_dict, fallback):
        """Extract script dict from various LLM response formats."""
        try:
            if isinstance(response_dict, dict) and "script" in response_dict:
                return response_dict["script"]
        except Exception:
            pass

        # If response is already a dict-like script, use it
        if isinstance(response_dict, dict) and len(response_dict) > 1:
            return response_dict

        # Try to extract the first non-metadata value
        if isinstance(response_dict, dict):
            for k, v in response_dict.items():
                if isinstance(v, dict):
                    return v

        # If error, use fallback
        logger.debug("failed to extract script from response, using fallback")
        return fallback
