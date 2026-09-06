import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Any

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
    current_task: Optional[str] = None

class MessageBus:
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.parent_queue: asyncio.Queue = asyncio.Queue()
        self._lock = threading.Lock()

    def register_subagent(self, subagent_id: str):
        with self._lock:
            if subagent_id not in self.queues:
                self.queues[subagent_id] = asyncio.Queue()

    def unregister_subagent(self, subagent_id: str):
        with self._lock:
            self.queues.pop(subagent_id, None)

    async def send_to_subagent(self, subagent_id: str, message: Dict[str, Any]):
        with self._lock:
            queue = self.queues.get(subagent_id)
            if queue is None:
                raise ValueError(f"Subagent {subagent_id} not registered")
        await queue.put(message)

    async def send_to_parent(self, message: Dict[str, Any]):
        await self.parent_queue.put(message)

    async def receive_from_parent(self) -> Dict[str, Any]:
        return await self.parent_queue.get()

class SubagentSpawner:
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.subagents: Dict[str, Subagent] = {}

    def spawn(self, role: SubagentRole) -> Subagent:
        subagent = Subagent(role=role)
        with self.message_bus._lock:
            self.subagents[subagent.id] = subagent
        self.message_bus.register_subagent(subagent.id)
        return subagent

    def get_subagent(self, subagent_id: str) -> Optional[Subagent]:
        with self.message_bus._lock:
            return self.subagents.get(subagent_id)

    async def assign_task(self, subagent_id: str, task: str):
        with self.message_bus._lock:
            subagent = self.subagents.get(subagent_id)
            if not subagent:
                raise ValueError(f"Subagent {subagent_id} not found")
        await self.message_bus.send_to_subagent(subagent_id, {"type": "task", "content": task})
        with self.message_bus._lock:
            current = self.subagents.get(subagent_id)
            if current is None:
                raise ValueError(f"Subagent {subagent_id} not found")
            current.current_task = task
            current.status = SubagentStatus.RUNNING

    async def terminate(self, subagent_id: str):
        with self.message_bus._lock:
            if subagent_id not in self.subagents:
                return
        try:
            await self.message_bus.send_to_subagent(subagent_id, {"type": "terminate"})
        except ValueError:
            with self.message_bus._lock:
                self.subagents.pop(subagent_id, None)
            return
        with self.message_bus._lock:
            current = self.subagents.get(subagent_id)
            if current is not None:
                current.status = SubagentStatus.DONE
        await asyncio.sleep(0.1)
        for _ in range(10):
            with self.message_bus._lock:
                queue = self.message_bus.queues.get(subagent_id)
                if queue is None:
                    break
                drained = queue.empty()
            if drained:
                break
            await asyncio.sleep(0.05)
        self.message_bus.unregister_subagent(subagent_id)
        with self.message_bus._lock:
            self.subagents.pop(subagent_id, None)
