from core.llm import llm, MBTIOutputParser
from core.prompts import get_axis_prompt_templates

class Chain:
    def __init__(self):
        self.templates = get_axis_prompt_templates()
        self.chains = {
            axis: (self.templates[axis] | llm | MBTIOutputParser())
            for axis in self.templates
        }

    async def arun(self, axis: str, user_feed: str, current_score: int, examples: str, missions: str) -> dict:
        prompt = self.templates[axis].format(
            user_feed=user_feed,
            current_score=current_score,
            examples=examples,
            missions=missions
        )
        raw = await llm.ainvoke(prompt)

        parsed = await self.chains[axis].ainvoke({
            "user_feed": user_feed,
            "current_score": current_score,
            "examples": examples,
            "missions": missions
        })
        return {"raw": raw, "parsed": parsed}