
from trpc.log import logger

def check_proposals(ctx, story_info, generate_input, proposals):
    # 检查参数
    if len(proposals) != generate_input.season_nums + 1:
        logger.error_context(
            ctx, f"proposals length: {len(proposals)} is not equal to {generate_input.season_nums + 1}")
        raise ValueError(f"proposals length: {len(proposals)} is not equal to {generate_input.season_nums + 1}")