STORY_ARC_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start_chapter_num": {"type": "integer"},
            "end_chapter_num": {"type": "integer"},
            "title": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["start_chapter_num", "end_chapter_num", "title", "content"],
    },
}

STORY_EVENT_SCHEMA = {
    "type": "object",
    "required": ["arc_title", "arc_summary", "events"],
    "properties": {
        "arc_title": {"type": "string"},
        "arc_summary": {"type": "string"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "title",
                    "chapter_range",
                    "summary",
                    "characters",
                    "conflict",
                    "emotion",
                    "narration",
                    "is_high_pressure",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "chapter_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "summary": {"type": "string"},
                    "characters": {"type": "array", "items": {"type": "string"}},
                    "conflict": {"type": "string"},
                    "emotion": {"type": "string"},
                    "narration": {"type": "string"},
                    "is_high_pressure": {"type": "boolean"},
                },
            },
        },
    },
}


DETAIL_OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "outline": {
            "type": "string",
            "description": "完整的分集故事摘要，纯文本格式",
        }
    },
    "required": ["outline"],
    "additionalProperties": False,
}

SCENE_PLAN_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["title", "content"],
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
        },
        "additionalProperties": False,
    },
}

SINGLE_SCENE_PLAN_SCHEMA = {
    "type": "object",
    "required": ["title", "content"],
    "properties": {
        "title": {"type": "string"},
        "content": {"type": "string"},
    },
    "additionalProperties": False,
}


def validate_json_schema(json_data, schema):
    import json
    import json_repair
    from jsonschema import validate, ValidationError

    try:
        # 如果输入是字符串，尝试解析为JSON
        if isinstance(json_data, str):
            json_data = json_repair.loads(json_data)
        if isinstance(schema, str):
            schema = json_repair.loads(schema)
        # 执行验证
        validate(instance=json_data, schema=schema)
        json_str = json.dumps(json_data, ensure_ascii=False)
        return True, json_str

    except json.JSONDecodeError as e:
        return False, f"JSON解析错误: {str(e)}"
    except ValidationError as e:
        return False, f"Schema验证失败: {str(e)}"
    except Exception as e:
        return False, f"验证过程中发生错误: {str(e)}"
