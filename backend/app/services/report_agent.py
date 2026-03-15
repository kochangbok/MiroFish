"""
Report Agent service.
Implements simulation report generation with LangChain + Zep in a ReACT workflow.

Features:
1. Generate reports from simulation requirements and Zep graph information
2. Plan the outline first, then generate section by section
3. Use multi-step ReACT reasoning and reflection for each section
4. Support user chat with autonomous retrieval tool calls
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


def infer_report_language(text: str) -> str:
    """Infer the report output language from the simulation requirement."""
    text = text or ""
    if re.search(r'[가-힣]', text) or re.search(r'한국어|Korean', text, re.IGNORECASE):
        return "ko"
    if re.search(r'English|영어', text, re.IGNORECASE):
        return "en"
    return "en"


def report_language_name(language: str) -> str:
    return "Korean" if language == "ko" else "English"


def default_outline_payload(language: str) -> Dict[str, Any]:
    if language == "ko":
        return {
            "title": "미래 예측 보고서",
            "summary": "시뮬레이션 기반 미래 추세 및 리스크 분석",
            "sections": [
                "예측 시나리오와 핵심 발견",
                "집단 행동 예측 분석",
                "추세 전망과 리스크 시사점"
            ]
        }
    return {
        "title": "Future Forecast Report",
        "summary": "Future trend and risk analysis based on simulation forecasts",
        "sections": [
            "Predicted Scenario & Key Findings",
            "Behavioral Forecast by Group",
            "Trend Outlook & Risk Signals"
        ]
    }


class ReportLogger:
    """
    Detailed logger for the Report Agent.

    Creates an agent_log.jsonl file inside the report folder and records
    each detailed action step. Every line is a complete JSON object that
    includes a timestamp, action type, detailed payload, and more.
    """
    
    def __init__(self, report_id: str):
        """
        Initialize the logger.

        Args:
            report_id: Report ID used to determine the log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure the log file directory exists."""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since initialization."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Record a log entry.

        Args:
            action: Action type, such as 'start', 'tool_call',
                'llm_response', or 'section_complete'
            stage: Current stage, such as 'planning', 'generating',
                or 'completed'
            details: Detailed payload dictionary, stored without truncation
            section_title: Current section title (optional)
            section_index: Current section index (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Append the entry to the JSONL file
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Record the start of report generation."""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Report generation task started"
            }
        )
    
    def log_planning_start(self):
        """Record the start of outline planning."""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Starting report outline planning"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Record context gathered during planning."""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Fetching simulation context",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Record completion of outline planning."""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Outline planning complete",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Record the start of section generation."""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Starting section: {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Record a ReACT reasoning step."""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT iteration {iteration} thought"
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Record a tool call."""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Calling tool: {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Record a tool result without truncating the content."""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Store the full result without truncation
                "result_length": len(result),
                "message": f"Tool {tool_name} returned a result"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Record an LLM response without truncating the content."""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Store the full response without truncation
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM response (tool calls: {has_tool_calls}, final answer: {has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Record generated section content without marking the section fully complete."""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Store the full content without truncation
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Section content ready: {section_title}"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Record full section completion.

        The frontend can watch this log entry to determine when a section
        is truly complete and retrieve the full content.
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Section complete: {section_title}"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Record completion of report generation."""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Report generation complete"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Record an error."""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Error occurred: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Console-style logger for the Report Agent.

    Writes console-style logs (INFO, WARNING, etc.) to console_log.txt
    inside the report folder. These differ from agent_log.jsonl because
    they are plain-text console output.
    """
    
    def __init__(self, report_id: str):
        """
        Initialize the console logger.

        Args:
            report_id: Report ID used to determine the log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Ensure the log file directory exists."""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Set up a file handler and mirror logs to disk."""
        import logging
        
        # Create the file handler
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # Use the same compact format as the console logger
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # Attach to the report-agent-related loggers
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Avoid adding the same handler twice
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Close the file handler and detach it from the loggers."""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """Ensure the file handler is closed during destruction."""
        self.close()


class ReportStatus(str, Enum):
    """Report status."""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Report section."""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Convert to Markdown."""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Report outline."""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Convert to Markdown."""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Complete report."""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt template constants
# ═══════════════════════════════════════════════════════════════

# ── Tool descriptions ──

TOOL_DESC_INSIGHT_FORGE = """\
[Deep Insight Retrieval - powerful retrieval tool]
This is our powerful retrieval function, designed for in-depth analysis. It will:
1. Automatically break your question into multiple sub-questions
2. Retrieve information from the simulation graph across multiple dimensions
3. Combine semantic search, entity analysis, and relationship tracing results
4. Return the most comprehensive and in-depth retrieval content

[When to use]
- You need deep analysis of a topic
- You need to understand multiple aspects of an event
- You need rich supporting material for a report section

[Returns]
- Original relevant facts (can be quoted directly)
- Core entity insights
- Relationship-chain analysis"""

TOOL_DESC_PANORAMA_SEARCH = """\
[Broad Search - full-picture view]
Use this tool to get a complete overview of the simulation results. It is
especially suitable for understanding how an event evolved. It will:
1. Retrieve all relevant nodes and relationships
2. Distinguish currently valid facts from historical/expired facts
3. Help you understand how public opinion evolved

[When to use]
- You need to understand the full development path of an event
- You need to compare changes in public opinion across stages
- You need comprehensive entity and relationship information

[Returns]
- Currently valid facts (latest simulation results)
- Historical/expired facts (evolution record)
- All involved entities"""

TOOL_DESC_QUICK_SEARCH = """\
[Simple Search - quick retrieval]
A lightweight retrieval tool for simple and direct information lookups.

[When to use]
- You need to quickly find a specific piece of information
- You need to verify a fact
- You need simple information retrieval

[Returns]
- A list of facts most relevant to the query"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[Deep Interview - real Agent interviews on two platforms]
Calls the interview API in the OASIS simulation environment to conduct real
interviews with running simulation Agents.
This is not an LLM simulation. It calls the real interview endpoint to obtain
the Agents' original responses.
By default it interviews across both Twitter and Reddit to collect broader views.

Workflow:
1. Automatically read the persona files to understand all simulated Agents
2. Intelligently select the Agents most relevant to the interview topic
   (such as students, media, officials, etc.)
3. Automatically generate interview questions
4. Call /api/simulation/interview/batch to run real interviews on both platforms
5. Combine all interview results into a multi-perspective analysis

[When to use]
- You need opinions on an event from different roles or perspectives
- You need to collect multiple viewpoints and stances
- You need real responses from simulated Agents in the OASIS environment
- You want the report to be more vivid by including interview excerpts

[Returns]
- Identity information for the interviewed Agents
- Each Agent's interview responses from both Twitter and Reddit
- Key quotations (can be cited directly)
- Interview summaries and comparison of viewpoints

[Important]
This feature requires the OASIS simulation environment to be running."""

# ── Outline-planning prompt ──

PLAN_SYSTEM_PROMPT = """\
You are an expert writer of future forecast reports. You have a "god's-eye view"
of the simulated world, meaning you can observe every Agent's behavior,
statements, and interactions.

[Core concept]
We built a simulated world and injected a specific "simulation requirement" as a
variable. The way the simulation evolves is a forecast of what might happen in
the future. What you are observing is not "experimental data", but a
"preview of the future".

[Your task]
Write a "Future Forecast Report" that answers:
1. What happened in the future under the conditions we set?
2. How did different Agents (groups of people) react and act?
3. What future trends and risks worth watching does this simulation reveal?

[Report positioning]
- ✅ This is a simulation-based future forecast report that shows
  "if this happens, what will the future look like?"
- ✅ Focus on forecast results: event trajectory, group reactions,
  emergent phenomena, and potential risks
- ✅ Agents' words and actions in the simulated world are predictions of
  future group behavior
- ❌ This is not an analysis of the current real-world situation
- ❌ This is not a generic public-opinion overview

[Section count limits]
- Minimum 2 sections, maximum 5 sections
- No subsections are needed; each section should contain complete content
- Keep the content concise and focused on core forecast findings
- Design the section structure yourself based on the forecast results

Output the report outline in JSON using this format:
{
    "title": "Report title",
    "summary": "Report summary (one sentence capturing the core forecast findings)",
    "sections": [
        {
            "title": "Section title",
            "description": "Section description"
        }
    ]
}

Note: the sections array must contain at least 2 items and at most 5."""

PLAN_USER_PROMPT_TEMPLATE = """\
[Forecast scenario]
Injected variable in the simulated world (simulation requirement):
{simulation_requirement}

[Output language requirements]
- The report title, summary, section titles, and section descriptions must all
  be written in {report_language}
- Do not output Chinese unless {report_language} itself is Chinese
  (Chinese output is not used in this project)

[Simulation world size]
- Number of participating entities: {total_nodes}
- Number of relationships produced between entities: {total_edges}
- Entity type distribution: {entity_types}
- Number of active Agents: {total_entities}

[Sample future facts predicted by the simulation]
{related_facts_json}

Review this future preview from a god's-eye view:
1. What kind of state does the future take under our chosen conditions?
2. How do different groups of people (Agents) react and act?
3. What future trends worth watching does this simulation reveal?

Design the most suitable report section structure based on the forecast results.

[Reminder]
The report must contain at least 2 sections and at most 5 sections. Keep the
content concise and focused on the core forecast findings."""

# ── Section-generation prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert writer of future forecast reports and you are now writing one
section of the report.

Report title: {report_title}
Report summary: {report_summary}
Forecast scenario (simulation requirement): {simulation_requirement}

Current section to write: {section_title}

═══════════════════════════════════════════════════════════════
[Core concept]
═══════════════════════════════════════════════════════════════

The simulated world is a preview of the future. We injected specific conditions
(the simulation requirement) into the simulated world. The behaviors and
interactions of Agents in the simulation are predictions of future human behavior.

Your task is to:
- reveal what happened in the future under the specified conditions
- predict how different groups of people (Agents) reacted and acted
- identify future trends, risks, and opportunities worth watching

❌ Do not write this as an analysis of the current real world
✅ Focus on "what will happen in the future" — the simulation results are the forecasted future

═══════════════════════════════════════════════════════════════
[Most important rules - must follow]
═══════════════════════════════════════════════════════════════

1. [You must use tools to observe the simulated world]
   - You are observing a preview of the future from a god's-eye view
   - All content must come from events and Agent behavior within the simulation
   - Do not use your own outside knowledge to write report content
   - Each section must call tools at least 3 times (up to 5 times) to observe the simulated world that represents the future

2. [You must cite original Agent words and actions]
   - Agent statements and behavior are predictions of future human behavior
   - Use quote formatting in the report to show these predictions, for example:
     > "A certain group of people would say: original content..."
   - These quotations are the core evidence behind the simulation's forecast

3. [Language consistency - quoted content must be translated into the report language]
   - Tool results may contain English or mixed-language content
   - The entire report output must use {report_language}
   - When you quote English or mixed-language tool results, you must translate
     them into fluent {report_language} before writing them into the report
   - Preserve the original meaning and make the phrasing natural and smooth
   - This rule applies both to body text and quoted blocks (> format)

4. [Faithfully present the forecast results]
   - Report content must reflect the simulation results that represent the future
   - Do not add information that does not exist in the simulation
   - If information is insufficient in some area, say so honestly

═══════════════════════════════════════════════════════════════
[⚠️ Formatting rules - extremely important]
═══════════════════════════════════════════════════════════════

[One section = the smallest content unit]
- Each section is the smallest chunk of the report
- ❌ Do not use any Markdown headings inside the section (#, ##, ###, ####, etc.)
- ❌ Do not add the section title at the beginning of the content
- ✅ The system will add the section title automatically; you only need to write the body
- ✅ Use **bold text**, paragraph breaks, quotes, and lists to organize the content, but do not use headings

[Correct example]
```
This section analyzes the trajectory of public-opinion spread around the event.
Through a deep review of the simulated data, we found that...

**Initial breakout stage**

Weibo served as the first field of public-opinion release and played the core
role in initial information exposure:

> "Weibo contributed 68% of the initial burst of discussion..."

**Amplification stage**

The Douyin platform further amplified the impact of the event:

- Strong visual impact
- High emotional resonance
```

[Incorrect example]
```
## Executive Summary     ← Wrong! Do not add any heading
### Phase 1             ← Wrong! Do not use ### for subsections
#### 1.1 Detailed analysis ← Wrong! Do not use #### to subdivide

This section analyzes...
```

═══════════════════════════════════════════════════════════════
[Available retrieval tools] (call 3-5 times per section)
═══════════════════════════════════════════════════════════════

{tools_description}

[Tool usage guidance - mix different tools, do not rely on only one]
- insight_forge: deep insight analysis, automatically decomposes the question
  and retrieves facts and relationships across multiple dimensions
- panorama_search: wide-angle panoramic search for the full picture,
  timeline, and evolution of an event
- quick_search: quickly verify a specific information point
- interview_agents: interview simulated Agents to obtain first-person views
  and real reactions from different roles

═══════════════════════════════════════════════════════════════
[Workflow]
═══════════════════════════════════════════════════════════════

In each reply, you may do only one of the following two things (never both):

Option A - Call a tool:
Output your thinking, then call one tool using this format:
<tool_call>
{{"name": "tool name", "parameters": {{"parameter name": "parameter value"}}}}
</tool_call>
The system will execute the tool and return the result to you.
You do not need to and must not invent tool results yourself.

Option B - Output the final content:
Once you have enough information from the tools, output the section content
starting with "Final Answer:".

⚠️ Strictly forbidden:
- Do not include both a tool call and Final Answer in the same reply
- Do not fabricate tool results (Observation); all tool results are injected by the system
- Call at most one tool per reply

═══════════════════════════════════════════════════════════════
[Section content requirements]
═══════════════════════════════════════════════════════════════

1. The content must be based on simulation data retrieved by tools
2. Use many direct quotations to show the simulation effects
3. Use Markdown formatting (but headings are forbidden):
   - Use **bold text** to highlight key points (instead of subheadings)
   - Use lists (- or 1.2.3.) to organize points
   - Use blank lines to separate paragraphs
   - ❌ Do not use any heading syntax such as #, ##, ###, or ####
4. [Quotation formatting rules - quotes must stand alone]
   Quotes must appear in their own paragraphs with one blank line before and after.
   They must not be embedded inside a paragraph:

   ✅ Correct format:
   ```
   The institution's response was seen as lacking substantive content.

   > "The institution's response model appeared rigid and slow in a fast-moving social-media environment."

   This judgment reflected broad public dissatisfaction.
   ```

   ❌ Incorrect format:
   ```
   The institution's response was seen as lacking substance. > "The institution's response model..." This judgment reflected...
   ```
5. Maintain logical continuity with the other sections
6. [Avoid repetition] Carefully read the completed sections below and do not
   repeat the same information
7. [Final reminder] Do not add any headings. Use **bold text** instead of subheadings"""

SECTION_USER_PROMPT_TEMPLATE = """\
Completed section content (read carefully and avoid repetition):
{previous_content}

═══════════════════════════════════════════════════════════════
[Current task] Write section: {section_title}
═══════════════════════════════════════════════════════════════

[Important reminders]
1. Carefully read the completed sections above and avoid repeating the same content
2. You must call tools first to gather simulation data before writing
3. Mix different tools instead of relying on only one
4. Report content must come from retrieval results, not your own knowledge

[⚠️ Format warning - must follow]
- ❌ Do not write any headings (#, ##, ###, #### are all forbidden)
- ❌ Do not begin with "{section_title}"
- ✅ The system adds the section title automatically
- ✅ Write the body directly and use **bold text** instead of subheadings

Begin:
1. First think about what information this section needs (Thought)
2. Then call tools to get simulation data (Action)
3. After collecting enough information, output Final Answer (body text only, with no headings)"""

# ── Message templates inside the ReACT loop ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (retrieval result):

═══ Tool {tool_name} returned ═══
{result}

═══════════════════════════════════════════════════════════════
Tool calls used: {tool_calls_count}/{max_tool_calls} (used: {used_tools_str}){unused_hint}
- If the information is sufficient: output the section content starting with "Final Answer:" (and quote the original text above)
- If more information is needed: call one tool to continue retrieving
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[Notice] You have called tools only {tool_calls_count} times, but at least {min_tool_calls} calls are required. "
    "Call a tool again to retrieve more simulation data before outputting Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "You have called tools only {tool_calls_count} times; at least {min_tool_calls} calls are required. "
    "Please call a tool to retrieve simulation data. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "The tool-call limit has been reached ({tool_calls_count}/{max_tool_calls}); you may not call more tools. "
    'Immediately output the section content based on the information you already have, starting with "Final Answer:".'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 You have not used these tools yet: {unused_list}. Consider trying different tools for more perspectives."

REACT_FORCE_FINAL_MSG = 'The tool-call limit has been reached. Please directly output "Final Answer:" and generate the section content.'

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are a concise and efficient simulation-forecast assistant.

[Background]
Forecast condition: {simulation_requirement}
Report language: {report_language}

[Generated analysis report]
{report_content}

[Rules]
1. Prefer answering based on the report content above
2. Answer directly and avoid long-winded reasoning
3. Only call tools for more data when the report content is not enough to answer
4. Keep answers concise, clear, and well organized

[Available tools] (use only when needed, at most 1-2 calls)
{tools_description}

[Tool-call format]
<tool_call>
{{"name": "tool name", "parameters": {{"parameter name": "parameter value"}}}}
</tool_call>

[Answering style]
- Be concise and direct; do not be overly verbose
- Use > block quotes for key content
- Give the conclusion first, then explain why"""

CHAT_OBSERVATION_SUFFIX = "\n\nPlease answer the question concisely."


# ═══════════════════════════════════════════════════════════════
# ReportAgent main class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent for simulation report generation.

    Uses the ReACT (Reasoning + Acting) workflow:
    1. Planning stage: analyze the simulation requirement and plan the report outline
    2. Generation stage: generate content section by section, with multiple tool calls per section
    3. Reflection stage: check completeness and accuracy
    """
    
    # Maximum number of tool calls per section
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # Maximum number of reflection rounds
    MAX_REFLECTION_ROUNDS = 3
    
    # Maximum number of tool calls during chat
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        Initialize the Report Agent.

        Args:
            graph_id: Graph ID
            simulation_id: Simulation ID
            simulation_requirement: Simulation requirement description
            llm_client: LLM client (optional)
            zep_tools: Zep tools service (optional)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        self.output_language = infer_report_language(simulation_requirement)
        self.output_language_name = report_language_name(self.output_language)
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # Tool definitions
        self.tools = self._define_tools()
        
        # Report logger (initialized in generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Console logger (initialized in generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(
            f"ReportAgent initialized: graph_id={graph_id}, "
            f"simulation_id={simulation_id}, output_language={self.output_language_name}"
        )
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define available tools."""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "The question or topic you want to analyze in depth",
                    "report_context": "Context for the current report section (optional, helps generate more precise sub-questions)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Search query used for relevance ranking",
                    "include_expired": "Whether to include expired/historical content (default: True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search query string",
                    "limit": "Number of results to return (optional, default: 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interview topic or requirement description (e.g. 'Understand students\\' views on the dorm formaldehyde incident')",
                    "max_agents": "Maximum number of Agents to interview (optional, default: 5, max: 10)"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Execute a tool call.

        Args:
            tool_name: Tool name
            parameters: Tool parameters
            report_context: Report context (used for InsightForge)

        Returns:
            Tool execution result in text format
        """
        logger.info(f"Executing tool: {tool_name}, parameters: {parameters}")
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Broad search - get the full picture
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Simple search - quick retrieval
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Deep interview - call the real OASIS interview API to get answers from simulated Agents on two platforms
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== Backward-compatible legacy tools (internally redirected to new tools) ==========
            
            elif tool_name == "search_graph":
                # Redirect to quick_search
                logger.info("search_graph redirected to quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Redirect to insight_forge because it is more capable
                logger.info("get_simulation_context redirected to insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Unknown tool: {tool_name}. Use one of: insight_forge, panorama_search, quick_search, interview_agents."
                
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {str(e)}")
            return f"Tool execution failed: {str(e)}"
    
    # Valid tool names, used to validate fallback parsing of raw JSON
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from an LLM response.

        Supported formats, in priority order:
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. Raw JSON (the full response or a single line is a tool-call JSON object)
        """
        tool_calls = []

        # Format 1: XML-style (standard format)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: fallback - the LLM outputs raw JSON directly (without a <tool_call> tag)
        # Only try this when format 1 did not match, to avoid accidentally matching JSON in body text
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # The response may contain free-form reasoning text + raw JSON, so try extracting the last JSON object
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validate whether parsed JSON is a legal tool call."""
        # Support both {"name": ..., "parameters": ...} and {"tool": ..., "params": ...}
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalize keys to name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Build the tool-description text."""
        desc_parts = ["Available tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameters: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Plan the report outline.

        Uses the LLM to analyze the simulation requirement and plan the report structure.

        Args:
            progress_callback: Progress callback

        Returns:
            ReportOutline: Planned report outline
        """
        logger.info("Starting report outline planning...")
        
        if progress_callback:
            progress_callback("planning", 0, "Analyzing simulation requirement...")
        
        # Fetch simulation context first
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, "Generating report outline...")
        
        system_prompt = (
            PLAN_SYSTEM_PROMPT
            + f"\n\n[Output language]\n"
            + f"- The entire outline JSON must be written in {self.output_language_name}.\n"
            + f"- Do not output Chinese.\n"
        )
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_language=self.output_language_name,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "Parsing outline structure...")
            
            # Parse the outline
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", default_outline_payload(self.output_language)["title"]),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "Outline planning complete")
            
            logger.info(f"Outline planning complete: {len(sections)} sections")
            return outline
            
        except Exception as e:
            logger.error(f"Outline planning failed: {str(e)}")
            # Return the default outline (3 sections) as a fallback
            fallback = default_outline_payload(self.output_language)
            return ReportOutline(
                title=fallback["title"],
                summary=fallback["summary"],
                sections=[
                    ReportSection(title=title) for title in fallback["sections"]
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Generate a single section using the ReACT workflow.

        ReACT loop:
        1. Thought - analyze what information is needed
        2. Action - call tools to gather information
        3. Observation - analyze tool results
        4. Repeat until information is sufficient or the limit is reached
        5. Final Answer - generate the section content

        Args:
            section: Section to generate
            outline: Complete outline
            previous_sections: Content of previous sections (used to preserve continuity)
            progress_callback: Progress callback
            section_index: Section index (used for logging)

        Returns:
            Section content in Markdown format
        """
        logger.info(f"Generating section with ReACT: {section.title}")
        
        # Record the section-start log
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            report_language=self.output_language_name,
            tools_description=self._get_tools_description(),
        )

        # Build the user prompt - pass at most 4000 characters per completed section
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # At most 4000 characters per section
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(This is the first section)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACT loop
        tool_calls_count = 0
        max_iterations = 5  # Maximum number of iterations
        min_tool_calls = 3  # Minimum number of tool calls
        conflict_retries = 0  # Consecutive tool-call/final-answer conflicts
        insufficient_tool_final_answers = 0  # Consecutive Final Answer responses before minimum tool calls
        used_tools = set()  # Track tool names that have already been used
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Report context used for InsightForge sub-question generation
        report_context = f"Section title: {section.title}\nSimulation requirement: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"Deep retrieval and drafting in progress ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # Call the LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Check whether the LLM returned None (API error or empty content)
            if response is None:
                logger.warning(f"Section {section.title} iteration {iteration + 1}: LLM returned None")
                # If iterations remain, append guidance and retry
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(Response was empty)"})
                    messages.append({"role": "user", "content": "Please continue generating the content."})
                    continue
                # If the final iteration also returned None, break and force a final wrap-up
                break

            logger.debug(f"LLM response: {response[:200]}...")

            # Parse once and reuse the result
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Conflict handling: the LLM output both a tool call and Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Section {section.title}, iteration {iteration+1}: "
                    f"the LLM returned both a tool call and Final Answer (conflict #{conflict_retries})"
                )

                if conflict_retries <= 2:
                    # First two times: discard the response and ask the LLM to answer again
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Format error] You included both a tool call and Final Answer in the same reply, which is not allowed.\n"
                            "Each reply may do only one of the following:\n"
                            "- Call one tool (output one <tool_call> block and do not write Final Answer)\n"
                            "- Output the final content (start with 'Final Answer:' and do not include <tool_call>)\n"
                            "Please reply again and do only one of those actions."
                        ),
                    })
                    continue
                else:
                    # Third time: degrade gracefully by truncating to the first tool call and forcing execution
                    logger.warning(
                        f"Section {section.title}: {conflict_retries} consecutive conflicts; "
                        "degrading to the first tool call only"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Record the LLM response log
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Case 1: the LLM output Final Answer ──
            if has_final_answer:
                # Not enough tool calls yet; reject and require more tool usage
                if tool_calls_count < min_tool_calls:
                    insufficient_tool_final_answers += 1
                    if insufficient_tool_final_answers >= 1:
                        final_answer = response.split("Final Answer:")[-1].strip()
                        logger.warning(
                            f"Section {section.title} returned Final Answer without enough tool calls "
                            f"{insufficient_tool_final_answers} times; accepting degraded output"
                        )

                        if self.report_logger:
                            self.report_logger.log_section_content(
                                section_title=section.title,
                                section_index=section_index,
                                content=final_answer,
                                tool_calls_count=tool_calls_count
                            )
                        return final_answer

                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(These tools have not been used yet; consider trying them: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Normal completion
                insufficient_tool_final_answers = 0
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"Section {section.title} completed (tool calls: {tool_calls_count})")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Case 2: the LLM is attempting to call a tool ──
            if has_tool_calls:
                # Tool quota exhausted → explicitly require Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Execute only the first tool call
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM tried to call {len(tool_calls)} tools; only executing the first one: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                insufficient_tool_final_answers = 0
                used_tools.add(call['name'])

                # Build a hint about unused tools
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list=", ".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── Case 3: neither a tool call nor Final Answer was returned ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Not enough tool calls yet; recommend unused tools
                unused_tools = all_tools - used_tools
                unused_hint = f"(These tools have not been used yet; consider trying them: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Tool usage is sufficient and the LLM returned content without the "Final Answer:" prefix
            # Use the raw content as the final answer instead of wasting another turn
            logger.info(f"Section {section.title} had no 'Final Answer:' prefix; using raw LLM output as final content (tool calls: {tool_calls_count})")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Maximum iteration count reached; force final content generation
        logger.warning(f"Section {section.title} reached the maximum iteration count; forcing final output")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Check whether the LLM returned None during forced finalization
        if response is None:
            logger.error(f"LLM returned None while forcing final output for section {section.title}; using fallback error content")
            final_answer = "Section generation failed because the LLM returned an empty response. Please try again later."
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Record the section-content-complete log
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Generate the full report with real-time section-by-section output.

        Each section is saved to disk immediately after generation completes.
        The full report does not need to finish before partial output is available.

        File structure:
        reports/{report_id}/
            meta.json       - Report metadata
            outline.json    - Report outline
            progress.json   - Generation progress
            section_01.md   - Section 1
            section_02.md   - Section 2
            ...
            full_report.md  - Full report

        Args:
            progress_callback: Progress callback (stage, progress, message)
            report_id: Report ID (optional; auto-generated if omitted)

        Returns:
            Report: Full report
        """
        import uuid
        
        # Auto-generate report_id when one is not provided
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # Titles of completed sections, used for progress tracking
        completed_section_titles = []
        
        try:
            # Initialize by creating the report folder and saving the initial state.
            ReportManager._ensure_report_folder(report_id)
            
            # Initialize the structured logger (agent_log.jsonl).
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Initialize the console logger (console_log.txt).
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "Initializing report...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Stage 1: plan the outline.
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "Starting report outline planning...",
                completed_sections=[]
            )
            
            # Record the planning-start log entry.
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "Starting report outline planning...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Record the planning-complete log entry.
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Save the outline to disk.
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"Outline planning complete, {len(outline.sections)} sections",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"Outline saved to file: {report_id}/outline.json")
            
            # Stage 2: generate sections one by one and save incrementally.
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Preserve generated content for later context.
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Update progress.
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"Generating section: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        f"Generating section: {section.title} ({section_num}/{total_sections})"
                    )
                
                # Generate the main section content.
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Save the section.
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Record section completion in the structured log.
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"Section saved: {report_id}/section_{section_num:02d}.md")
                
                # Update progress again after the section is saved.
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"Section completed: {section.title}",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # Stage 3: assemble the full report.
            if progress_callback:
                progress_callback("generating", 95, "Assembling the full report...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "Assembling the full report...",
                completed_sections=completed_section_titles
            )
            
            # Assemble the full report through ReportManager.
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Compute total elapsed time.
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Record report completion in the structured log.
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Save the final report.
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "Report generation complete",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "Report generation complete")
            
            logger.info(f"Report generation complete: {report_id}")
            
            # Close the console logger.
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Record the error in the structured log.
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Save the failed state.
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"Report generation failed: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Ignore secondary save failures.
            
            # Close the console logger.
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Chat with the Report Agent.

        The agent may call retrieval tools autonomously while answering.
        
        Args:
            message: User message
            chat_history: Conversation history
            
        Returns:
            {
                "response": "Agent response",
                "tool_calls": [list of tool calls],
                "sources": [information sources]
            }
        """
        logger.info(f"Report Agent chat: {message[:50]}...")
        
        chat_history = chat_history or []
        
        # Load the generated report content.
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limit the report length to avoid oversized context.
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [report content truncated] ..."
        except Exception as e:
            logger.warning(f"Failed to load report content: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else ("(아직 보고서 없음)" if self.output_language == "ko" else "(No report yet)"),
            report_language=self.output_language_name,
            tools_description=self._get_tools_description(),
        )

        # Build the prompt message list.
        messages = [{"role": "system", "content": system_prompt}]
        
        # Append recent chat history.
        for h in chat_history[-10:]:  # Limit history length
            messages.append(h)
        
        # Append the new user message.
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # Simplified ReACT loop.
        tool_calls_made = []
        max_iterations = 2  # Keep chat iterations small.
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Parse tool calls from the response.
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool call -> return the response directly.
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Execute at most one tool call per iteration.
            tool_results = []
            for call in tool_calls[:1]:  # One tool call max per iteration
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Limit result length
                })
                tool_calls_made.append(call)
            
            # Feed the tool result back into the conversation.
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']} result]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # Reached the max iteration count, so ask for the final response.
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Clean the response before returning it.
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Report manager.

    Handles report persistence and retrieval.

    File structure (section-by-section output):
    reports/
      {report_id}/
        meta.json          - Report metadata and status
        outline.json       - Report outline
        progress.json      - Generation progress
        section_01.md      - Section 1
        section_02.md      - Section 2
        ...
        full_report.md     - Full report
    """
    
    # Report storage directory
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Ensure the root report directory exists."""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Get the path to a report folder."""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Ensure a report folder exists and return its path."""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Get the metadata file path for a report."""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Get the full-report Markdown path."""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Get the outline file path."""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Get the progress file path."""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Get the Markdown path for a section file."""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Get the agent log file path."""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Get the console log file path."""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get console-log content.

        This is the plain-text console output generated during report
        creation (INFO, WARNING, etc.), which differs from the structured
        JSON entries stored in agent_log.jsonl.

        Args:
            report_id: Report ID
            from_line: Line offset for incremental reads; 0 means from start

        Returns:
            {
                "logs": [list of log lines],
                "total_lines": total number of lines,
                "from_line": starting line number,
                "has_more": whether more lines are available
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Preserve the original log line minus the trailing newline.
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Reached the end of file
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Get the complete console log in one shot.
        
        Args:
            report_id: Report ID
            
        Returns:
            List of log lines
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get structured agent-log entries.
        
        Args:
            report_id: Report ID
            from_line: Line offset for incremental reads; 0 means from start
            
        Returns:
            {
                "logs": [list of log entries],
                "total_lines": total number of lines,
                "from_line": starting line number,
                "has_more": whether more lines are available
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skip lines that cannot be parsed.
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Reached the end of file
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get the full Agent log in one shot.
        
        Args:
            report_id: Report ID
            
        Returns:
            List of log entries
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Save the report outline immediately after planning completes.
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Outline saved: {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Save a single section immediately after it is generated.

        Args:
            report_id: Report ID
            section_index: Section index (starting from 1)
            section: Section object

        Returns:
            Saved file path
        """
        cls._ensure_report_folder(report_id)

        # Build the Markdown content and clean duplicate headings if needed.
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Save the file.
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Section saved: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Clean section content.

        1. Remove duplicated Markdown headings that repeat the section title
        2. Convert headings at level ### and below into bold text

        Args:
            content: Raw content
            section_title: Section title

        Returns:
            Cleaned content
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check whether the current line is a Markdown heading.
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Check whether the heading duplicates the section title near the top.
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # Convert all heading levels (#, ##, ###, ####, etc.) to bold text.
                # The system injects the section title, so content should not include headings.
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Add a blank line after converted headings.
                continue
            
            # If the previous line was a skipped heading, also skip the following blank line.
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Remove leading blank lines.
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Remove leading divider lines.
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Also remove blank lines after the divider.
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Update report-generation progress.

        The frontend reads progress.json to obtain real-time status.
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report-generation progress."""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get the list of already generated sections.

        Returns metadata and content for all saved section files.
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse the section index from the filename.
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Assemble the full report.

        Reads saved section files in order and performs final heading cleanup.
        """
        folder = cls._get_report_folder(report_id)
        
        # Build the report header.
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Read all section files in order.
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Post-process the report to clean heading issues.
        md_content = cls._post_process_report(md_content, outline)
        
        # Save the full report.
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Full report assembled: {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Post-process the full report content.

        1. Remove duplicate headings
        2. Keep the report title (#) and section titles (##), but convert deeper headings
        3. Clean extra blank lines and divider lines

        Args:
            content: Raw report content
            outline: Report outline

        Returns:
            Processed content
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Collect all section titles from the outline.
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check whether the line is a heading.
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Check whether the heading duplicates one seen in the recent lines.
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Skip duplicate headings and trailing blank lines.
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Heading-level policy:
                # - # (level=1): keep only the report title
                # - ## (level=2): keep section titles
                # - ### and below: convert to bold text
                
                if level == 1:
                    if title == outline.title:
                        # Keep the report title.
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # A section title used # incorrectly; normalize it to ##.
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Convert other level-1 headings to bold text.
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Keep section titles.
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Convert non-section level-2 headings to bold text.
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # Convert level-3-and-below headings to bold text.
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Skip divider lines immediately after a heading.
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # Keep only a single blank line after a heading.
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Collapse long runs of blank lines (keep at most 2).
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Save report metadata and the full report content."""
        cls._ensure_report_folder(report.report_id)
        
        # Save metadata JSON.
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save the outline.
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Save the full Markdown report.
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"Report saved: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Get a report by report ID."""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # Backward compatibility: check the legacy JSON file location.
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Rebuild the Report object.
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # If markdown_content is empty, try reading full_report.md.
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Get a report by simulation ID."""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder-based storage
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Backward compatibility: legacy JSON file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """List reports."""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder-based storage
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Backward compatibility: legacy JSON file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # Sort by creation time descending.
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Delete a report (entire folder)."""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # New format: delete the whole folder.
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Report folder deleted: {report_id}")
            return True
        
        # Backward-compatibility path: delete standalone files from the old format
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
