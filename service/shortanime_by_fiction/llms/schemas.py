import json
from jsonschema import validate, ValidationError
from jsonschema.validators import Draft7Validator

def create_proposal_schema(max_plot_id=1000):
    """
    根据传入的max_plot_id动态生成schema
    
    Args:
        max_plot_id (int): plot_id的最大值，默认为1000
        
    Returns:
        dict: 动态生成的JSON schema
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "plot_planning": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "plot_id": {
                            "type": "integer", 
                            "minimum": 0,
                            "maximum": max_plot_id
                        },
                        "status": {"type": "string", "enum": ["保留", "删除"]},
                        "reason": {"type": "string"}
                    },
                    "required": ["plot_id", "status"]
                }
            },
            "point_planning": {
                "type": "object",
                "properties": {
                    "opening": {"$ref": "#/definitions/plot_range"},
                    "first_three_eps": {"$ref": "#/definitions/plot_range"},
                    "first_paywall": {"$ref": "#/definitions/plot_range"},
                    "season_finale": {"$ref": "#/definitions/plot_range"}
                }
            },
            "story_lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "status": {"type": "string", "enum": ["保留", "精简", "删除"]},
                        "detailed_story_lines": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_chapter_id": {"type": "integer", "minimum": 0},
                                    "end_chapter_id": {"type": "integer", "minimum": 0},
                                    "title": {"type": "string"},
                                    "content": {"type": "string"}
                                },
                                "required": ["start_chapter_id", "end_chapter_id"]
                            }
                        }
                    },
                    "required": ["name", "status"]
                }
            },
            "other_adaptation_proposal": {"type": "string"}
        },
        "definitions": {
            "plot_range": {
                "type": "object",
                "properties": {
                    "start_plot_id": {
                        "type": "integer", 
                        "minimum": 0,
                        "maximum": max_plot_id
                    },
                    "end_plot_id": {
                        "type": "integer", 
                        "minimum": 0,
                        "maximum": max_plot_id
                    },
                    "description": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["start_plot_id", "end_plot_id"]
            }
        }
    }


def validate_proposal_data(data, max_plot_id=1000):
    """
    验证提案数据，使用动态生成的schema
    
    Args:
        data (dict): 要验证的数据
        max_plot_id (int): plot_id的最大值
        
    Returns:
        bool: 验证是否通过
        
    Raises:
        ValidationError: 如果验证失败
    """
    schema = create_proposal_schema(max_plot_id)
    validate(data, schema)
    return True


# 保留原有的静态schema定义，但标记为已弃用
proposal_schema_definition = create_proposal_schema(1000)