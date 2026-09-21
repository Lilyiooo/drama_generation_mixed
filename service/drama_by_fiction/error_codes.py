class BizCode:
    PARAM_INVALID = 10001
    USER_NOT_FOUND = 10002
    NO_PERMISSION = 10003
    ITEM_NOT_FOUND = 10004
    DUPLICATE = 10005
    INTERNAL_ERROR = 19999


BIZ_MSG = {
    BizCode.PARAM_INVALID: "参数不合法",
    BizCode.USER_NOT_FOUND: "用户不存在",
    BizCode.NO_PERMISSION: "无权限",
    BizCode.ITEM_NOT_FOUND: "资源不存在",
    BizCode.DUPLICATE: "数据已存在",
    BizCode.INTERNAL_ERROR: "内部错误",
}
