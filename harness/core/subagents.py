import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any

class SubagentRole(Enum):
    RESEARCHER = auto()
    CODER = auto()
    VERIFIER = auto()

class SubagentStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    DONE = auto()
    ERROR = auto()

@dataclass
class Subagent:
    role: SubagentRole
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: SubagentStatus = SubagentStatus.IDLE
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    current_task: Optional[str] = None

class MessageBus:
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.parent_queue: asyncio.Queue = asyncio.Queue()

    def register_subagent(self, subagent_id: str):
        if subagent_id not in self.queues:
            self.queues[subagent_id] = asyncio.Queue()

    def unregister_subagent(self, subagent_id: str):
        if subagent_id in self.queues:
            del self.queues[subagent_id]

    async def send_to_subagent(self, subagent_id: str, message: Dict[str, Any]):
        if subagent_id in self.queues:
            await self.queues[subagent_id].put(message)
        else:
            raise ValueError(f"Subagent {subagent_id} not registered")

    async def send_to_parent(self, message: Dict[str, Any]):
        await self.parent_queue.put(message)

    async def receive_from_subagent(self, subagent_id: str) -> Dict[str, Any]:
        if subagent_id in self.queues:
            return await self.queues[subagent_id].get()
        raise ValueError(f"Subagent {subagent_id} not registered")

    async def receive_from_parent(self) -> Dict[str, Any]:
        return await self.parent_queue.get()

class SubagentSpawner:
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.subagents: Dict[str, Subagent] = {}

    def spawn(self, role: SubagentRole) -> Subagent:
        subagent = Subagent(role=role)
        self.subagents[subagent.id] = subagent
        self.message_bus.register_subagent(subagent.id)
        return subagent

    def get_subagent(self, subagent_id: str) -> Optional[Subagent]:
        return self.subagents.get(subagent_id)

    async def assign_task(self, subagent_id: str, task: str):
        subagent = self.get_subagent(subagent_id)
        if not subagent:
            raise ValueError(f"Subagent {subagent_id} not found")
        
        subagent.current_task = task
        subagent.status = SubagentStatus.RUNNING
        await self.message_bus.send_to_subagent(subagent_id, {"type": "task", "content": task})

    async def terminate(self, subagent_id: str):
        subagent = self.get_subagent(subagent_id)
        if not subagent:
            return
        subagent.status = SubagentStatus.DONE
        await self.message_bus.send_to_subagent(subagent_id, {"type": "terminate"})
        self.message_bus.unregister_subagent(subagent_id)
        del self.subagents[subagent_id]
