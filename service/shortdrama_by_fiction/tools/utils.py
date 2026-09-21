from copy import deepcopy as copy
import json
import time
import os
from trpc.log import logger
from service.shortdrama_by_fiction.data_model import ShortDramaInfo

from service.shortdrama_by_fiction.tools.llm_services import query_llm

def get_save_drama_info(short_drama_info):
    # 使用新的序列化方法
    #short_drama_info = ipdata_dict.to_short_drama_info()
    shortdrama_info = ShortDramaInfo(
        fname=short_drama_info.get('fname', ''),
        outline=short_drama_info.get('outline', {}),
        heros_str=short_drama_info.get('heros_str', ''),
        world_str=short_drama_info.get('world_str', ''),
        mainline=short_drama_info.get('mainline', ''),
        subline=short_drama_info.get('subline', ''),
        outline_plan_str=short_drama_info.get('outline_plan_str', ''),
        outline_plan=short_drama_info.get('outline_plan', {}),
        num_episode=short_drama_info.get('num_episode', 0)
    )
    return shortdrama_info 


class BaseResultOperation:
    def __init__(self):
        pass

class BaseLlmOperation:
    def __init__(self, cfg, stage='shortdrama_gen_pipeline',name="noname", retry_times=1):
        self.cfg = cfg 
        self.stage=stage
        self.name = name
        self.retry_times=retry_times
        self.bkmodel_name = "gemini-3-pro" # "deepseek-r1-local-II"

    def _decode_as_json(self, data, prompt):
        if isinstance(data, dict) and (data=={} or "error" in data or ("error" in data and "message" in data["error"] and "PROHIBITED_CONTENT" == data["error"]["message"])):
            return {}
        ori_data = copy(data)
        opt_times = 0
        valid_data = False
        while opt_times < self.retry_times:
            try:
                if "```" in data:
                    data = data.replace('```json', '').replace('```', '')
                data = json.loads(data)
                #data = eval(data)
                valid_data = True
                break
            except:
                print("======>llm decode: prompt", len(prompt),prompt[:50])
                print(f"优化{self.stage} {self.name}结果第{opt_times+1}次尝试")
                _,  data = query_llm(
                        prompt,
                        model_name=self.cfg.model_name,
                        model_type=self.cfg.model_type,
                )
                opt_times += 1
        if "```" in data:
            data = data.replace('```json', '').replace('```', '')
            data = json.loads(data)
            #data = eval(data)
            valid_data = True 
        if not valid_data:
            print(f"optimize not right{ori_data} -> {data}")
            return {}
        print("done decoding")
        return data

    def _gen_llm_proc(self, prompt, optim, task_type="", novel_id="00", isregen=False, suggest=""):
        response = {}
        if not os.path.exists(f"tmp/"):
            os.makedirs(f"tmp/")
        # os.path.exists("/cfs/cfs_dataV") and 
        if (not isregen) and \
                os.path.exists(f"tmp/{novel_id}_{task_type}{suggest}_response.txt"):
            try:
                with open(f"tmp/{novel_id}_{task_type}{suggest}_response.txt", "r") as f:
                    ss = f.read()
                    if "```" in ss:
                        ss = ss.replace('```json', '').replace('```', '')
                    response = json.loads(ss)
                    print("loading from local file", f"tmp/{novel_id}_{task_type}_response.txt", response)
                    return response
            
            except:
                print(f"!!!!!not valid results file tmp/{novel_id}_{task_type}_response.txt")
        for ii in range(self.cfg.retry_cnt):
            #if ii >= 2:
            #    time.sleep(10)
            try:
                print("-------llm proc")
                print("prompt: len/info",len(prompt),prompt[:100])
                print("____")
                _, response = query_llm(
                    prompt,
                    model_name=self.cfg.model_name,
                    model_type=self.cfg.model_type,
                )
                print("before optim response", response)
                optim_prompt = optim.replace(f"{{result}}", str(response))
                print("optim_prompt", optim_prompt)
                with open(f"tmp/{novel_id}_{task_type}{suggest}_response.txt", "w") as f:
                    f.write(response)
                response = self._decode_as_json(response, optim_prompt) 
                break
            except Exception as e:
                logger.error_context(
                    self.ctx,
                    f"base-LLM {novel_id}-{task_type} err{ii}: {e} ",
                )
            time.sleep(5)
        # fast fix with deepseek
        
        try:
            _, response = query_llm(
                prompt,
                model_name=self.bkmodel_name,
                model_type=self.cfg.model_type,
            )
            optim_prompt = optim.replace(f"{{result}}", str(response))
            response = self._decode_as_json(response, optim_prompt) 
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"backup-LLM {novel_id}-{task_type} err{ii}: {e} ")
        
        return response