import asyncio
from typing import Dict, Optional

class PipelineEventBus:
    def __init__(self):
        # Maps spill_id -> asyncio.Queue of event dicts
        self._queues: Dict[str, asyncio.Queue] = {}

    def subscribe(self, spill_id: str) -> asyncio.Queue:
        if spill_id not in self._queues:
            self._queues[spill_id] = asyncio.Queue()
        return self._queues[spill_id]

    def unsubscribe(self, spill_id: str):
        if spill_id in self._queues:
            del self._queues[spill_id]

    async def emit(self, spill_id: str, stage: str, percent: int, message: str):
        if spill_id in self._queues:
            await self._queues[spill_id].put({
                "spill_id": spill_id,
                "stage": stage,
                "percent": percent,
                "message": message
            })

bus = PipelineEventBus()
