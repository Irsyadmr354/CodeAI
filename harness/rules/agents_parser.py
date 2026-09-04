import re
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

@dataclass
class AgentRules:
    allowed: List[str] = field(default_factory=list)
    forbidden: List[str] = field(default_factory=list)
    boundaries: List[str] = field(default_factory=list)

class AgentsParser:
    """Parser for AGENTS.md that extracts boundary rules and provides pre-flight verification."""
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def parse(self) -> AgentRules:
        if not self.file_path.exists():
            return AgentRules()
        
        content = self.file_path.read_text(encoding='utf-8')
        rules = AgentRules()
        
        current_section = None
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            
            header_match = re.match(r'^#+\s+(Allowed|Forbidden|Boundaries)', line, re.IGNORECASE)
            if header_match:
                current_section = header_match.group(1).lower()
                continue
            
            if current_section and (line.startswith('- ') or line.startswith('* ')):
                item = line[2:].strip()
                if current_section == 'allowed':
                    rules.allowed.append(item)
                elif current_section == 'forbidden':
                    rules.forbidden.append(item)
                elif current_section == 'boundaries':
                    rules.boundaries.append(item)
                    
        return rules

    def verify_tool_call(self, tool_name: str, payload: dict, rules: AgentRules) -> bool:
        """
        Pre-flight verification: engine validates tool call payloads against AGENTS.md boundaries.
        Returns False if a forbidden keyword is found in the tool_name or payload values.
        """
        payload_str = str(payload).lower()
        tool_name = tool_name.lower()
        
        for forbidden in rules.forbidden:
            f_lower = forbidden.lower()
            if f_lower in tool_name or f_lower in payload_str:
                return False
                
        return True
