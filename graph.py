import asyncio
import os
import json
import copy
from typing import TypedDict, Annotated, Any
from collections import defaultdict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================================
# STATE DEFINITION
# ============================================================================

class PreDiscussionState(TypedDict):
    messages: Annotated[list, add_messages]
    raw_input: dict
    metadata: dict
    rfd_list: dict
    user_story_groups: dict
    
    latest_dev_output: list
    rfd_output: dict
    summary_highlights: dict
    
    UserStories: list
    RFDList: dict
    Summary_highlights: dict
    
    latest_dev_feedback: str
    rfd_feedback: str
    latest_dev_loop_count: int
    rfd_loop_count: int


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def init_chat_model(model: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4.1", temperature=0)


def parse_model_json(content: str) -> Any:
    content = content.strip()
    
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    
    content = content.strip()
    
    return json.loads(content)


# ============================================================================
# PREPROCESSING NODE
# ============================================================================

def preprocessing_node(state: PreDiscussionState) -> PreDiscussionState:
    
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages provided in state")
    
    last_message = messages[-1]
    
    # Handle both LangChain message objects and dict formats
    if hasattr(last_message, 'content'):
        message_content = last_message.content
    elif isinstance(last_message, dict):
        message_content = last_message.get("content", "")
    else:
        message_content = str(last_message)
    
    if isinstance(message_content, list):
        text_parts = []
        for item in message_content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        message_content = "".join(text_parts).strip()
    elif isinstance(message_content, bytes):
        message_content = message_content.decode("utf-8")
    
    if not message_content:
        raise ValueError("Message content is empty")
    
    metadata_input = {
        "board_name": "Unknown",
        "sprint_name": "Unknown",
        "board_category": "scrum"
    }
    rfd_list = {"risks": [], "followups": [], "dependencies": []}
    user_stories = []
    
    if "<Metadata>" in message_content and "</Metadata>" in message_content:
        metadata_str = message_content.split("<Metadata>")[1].split("</Metadata>")[0].strip()
        try:
            parsed_metadata = json.loads(metadata_str)
            metadata_input.update(parsed_metadata)
        except json.JSONDecodeError:
            pass
    
    if "<UserStories>" in message_content and "</UserStories>" in message_content:
        user_stories_str = message_content.split("<UserStories>")[1].split("</UserStories>")[0].strip()
        try:
            user_stories = json.loads(user_stories_str)
        except json.JSONDecodeError:
            pass
    
    if "<RFDList>" in message_content and "</RFDList>" in message_content:
        rfd_str = message_content.split("<RFDList>")[1].split("</RFDList>")[0].strip()
        try:
            rfd_list = json.loads(rfd_str)
        except json.JSONDecodeError:
            pass
    else:
        try:
            raw_input = json.loads(message_content)
            metadata_input.update(raw_input.get("metadata", {}))
            user_stories = raw_input.get("user_stories", [])
            rfd_list = raw_input.get("rfd_list", {"risks": [], "followups": [], "dependencies": []})
        except json.JSONDecodeError:
            if isinstance(message_content, dict):
                metadata_input.update(message_content.get("metadata", {}))
                user_stories = message_content.get("user_stories", [])
                rfd_list = message_content.get("rfd_list", {"risks": [], "followups": [], "dependencies": []})
    
    raw_input = {
        "metadata": metadata_input,
        "user_stories": user_stories,
        "rfd_list": rfd_list
    }
    
    user_story_groups = defaultdict(lambda: defaultdict(list))
    
    for story in user_stories:
        status = story.get("Status", "Unknown")
        assignee = story.get("Assignee", "Unassigned")
        user_story_groups[status][assignee].append(story)
    
    user_story_groups = {status: dict(assignees) for status, assignees in user_story_groups.items()}
    
    return {
        "messages": messages,
        "raw_input": raw_input,
        "metadata": metadata_input,
        "rfd_list": rfd_list,
        "user_story_groups": user_story_groups,
        "latest_dev_output": [],
        "rfd_output": {"risks": [], "followups": [], "dependencies": []},
        "summary_highlights": {},
        "latest_dev_feedback": "",
        "rfd_feedback": "",
        "latest_dev_loop_count": 0,
        "rfd_loop_count": 0,
        "UserStories": [],
        "RFDList": {"risks": [], "followups": [], "dependencies": []},
        "Summary_highlights": {}
    }


# ============================================================================
# LATEST DEV AGENT
# ============================================================================

async def latest_dev_agent(state: PreDiscussionState) -> PreDiscussionState:
    
    system_prompt = """<tasks>
<description>
you are the stories reasoning agent (scrum master), you are given:
A latest-snapshot of JIRA User Stories for the active sprint for a board.
A previously updated JSON of 3 arrays : risks, follow_ups, dependencies that hold previously identified Risks, Followups and Dependencies (RFD).
(keys:risks, follow_ups, dependencies; each item has the following keys: Story, followup/risk/dependency_detail, status, rule_id, update, source, open_date, close_date) .
</description>
<objectives>
If you receive instructions to fix stories from validation agent, prioritize that, otherwise skip this line.
Analyze the latest snapshot of JIRA User Stories to identify key themes and areas for improvement.
Cross-reference the identified risks, follow-ups, and dependencies with the latest user stories to ensure all aspects are covered.
Capture the comment section and worklog of the user stories, and make sure to pass them to the next Agent.
Pass forward all the details related to the user stories including the metadata, comments and worklogs to the next agent.
</objectives>
<outputs>
Latest Developments: for each team member and status add a field called latest_development. This field contain a summary of the latest updates of the member user stories based grouped for.
The output must have this tags with its content even if you didn't change anything, and it must be only this format <UserStories>[{'assignee': '', 'status': '','latest_development': '','Number':['','',...,'']}]
Make sure none of the stories/subtasks/bugs... are missing.
Make Sure none of the keys 'assignee', 'status','latest_development','Number'.
</outputs>
</tasks>

CRITICAL: Output ONLY a valid JSON array. No markdown, no explanations, no code blocks. Start with [ and end with ]"""

    user_input = {
        "user_story_groups": state.get("user_story_groups", {}),
        "rfd_list": state.get("rfd_list", {}),
        "feedback": state.get("latest_dev_feedback", "")
    }
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(user_input, indent=2))
    ]
    
    model = init_chat_model(model="gpt-4.1")
    response = model.invoke(messages)
    
    try:
        latest_dev_output = parse_model_json(response.content)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"ERROR: latest_dev_agent parsing failed")
        print(f"Raw response (first 500 chars): {response.content[:500]}")
        print(f"Error: {exc}")
        
        latest_dev_output = []
    
    if not isinstance(latest_dev_output, list):
        print(f"WARNING: latest_dev_output is not a list, converting. Type: {type(latest_dev_output)}")
        latest_dev_output = []
    
    new_loop_count = state.get("latest_dev_loop_count", 0) + 1
    
    return {
        "latest_dev_output": latest_dev_output,
        "latest_dev_loop_count": new_loop_count
    }


# ============================================================================
# RFD AGENT
# ============================================================================

async def rfd_agent(state: PreDiscussionState) -> PreDiscussionState:
    
    system_prompt = """<tasks>
<description>
You are the Risks, Follow ups and Dependencies (RFDs) agent, you are given:
A latest-snapshot of JIRA User Stories for the active sprint for a board.
A previously updated JSON of risks, followups, dependencies that hold previously identified Risks, Followups and Dependencies (RFD).
(keys:risks, follow_ups, dependencies; each item has the following keys: Story, followup/risk/dependency_detail, status, update, source, rule_id, open_date, close_date...).
</description>
<objectives>
1. Cross scan the RFDs input and the user stories to make sure all the RFDs are covered.
2. If there are no RFDs, rely on user stories to identify any new RFDs.
3. scan the snapshot (especially in the comments and worklog for each user story) for any new Risks , Follow-ups or Dependencies that are not present in the prior-day RFDs.
4. If the user story is moved to closed/completed or similar status, all related RFDs should be Closed.
5. If there is an RFD with a user story number that is not present in the user stories, it should be closed.
</objectives>
<outputs>
only output an updated list of RFDs like the input json, no extra text.
The only fields that should be updated on existing RFDs are: status, update, source, rule_id, close_date.
If a new RFD is identified, it should be added to the output list in the same format and all the fields must be populated except close_date.
rule_id should be null
Make sure all dates are following MM-DD-YYYY format.
Make sure story numbers are consistent between the RFDs and user stories.
Make sure source is reflecting the source of update, closure or creation of the RFD, the source can be multiple as well since its an array.
Make sure open_date is never empty.
Make sure risk/dependency/followup_detail is never null, it should have why is the item a risk/dependency/followup.
The output must have all the input tags with their content even if you didn't change them
subtasks key in the rfds should be a list of the issue numbers only. eg: "subtask": ["NUMBER-1", "NUMBER-2"]
</outputs>
<RFDList>updated content</RFDList>
</tasks>

CRITICAL: Output ONLY a valid JSON object. No markdown, no explanations, no code blocks. Start with { and end with }. Must have keys: risks, followups, dependencies"""

    user_input = {
        "rfd_list": state.get("rfd_list", {}),
        "user_story_groups": state.get("user_story_groups", {}),
        "feedback": state.get("rfd_feedback", "")
    }
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(user_input, indent=2))
    ]
    
    model = init_chat_model(model="gpt-4.1")
    response = model.invoke(messages)
    
    try:
        rfd_output = parse_model_json(response.content)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"ERROR: rfd_agent parsing failed")
        print(f"Raw response (first 500 chars): {response.content[:500]}")
        print(f"Error: {exc}")
        
        rfd_output = {"risks": [], "followups": [], "dependencies": []}
    
    if not isinstance(rfd_output, dict):
        print(f"WARNING: rfd_output is not a dict, converting. Type: {type(rfd_output)}")
        rfd_output = {"risks": [], "followups": [], "dependencies": []}
    
    for category in ["risks", "followups", "dependencies"]:
        if category not in rfd_output:
            rfd_output[category] = []
        for item in rfd_output[category]:
            if "issue" in item and "Story" not in item:
                item["Story"] = item.pop("issue")
    
    new_loop_count = state.get("rfd_loop_count", 0) + 1
    
    return {
        "rfd_output": rfd_output,
        "rfd_loop_count": new_loop_count
    }


# ============================================================================
# VALIDATION AGENT
# ============================================================================

async def validation_agent(state: PreDiscussionState) -> PreDiscussionState:
    
    system_prompt = """<tasks>
<description>
You are the Validation Agent you are given:
Updated RFD List.
Updated Stories with latest_development field.
metadata of the board.
</description>
<objectives>
Provides specific feedback for each agent or marks as complete.
</objectives>
<task>
Validate both outputs for:

**Latest Dev Output:**
- All status groups are covered
- All assignees within each status are covered
- Story numbers match exactly from the input user stories
- Summaries are factual and based on comments and worklogs
- No missing or hallucinated information

**RFD Output:**
- All existing RFDs are accounted for
- RFDs are properly closed when stories are completed
- New RFDs are justified from story comments/worklogs
- RFDs are closed when story numbers no exist
- Dates are in MM-DD-YYYY format
- open_date is never empty
- story numbers match exactly

</task>
<output_format>
Return a JSON object with this exact structure:
{
  "latest_dev_feedback": "complete" OR "specific feedback on what needs fixing",
  "rfd_feedback": "complete" OR "specific feedback on what needs fixing"
}
</output_format>
<feedback_rules>
- If output is correct and complete, return "complete"
- If there are issues, provide specific, ACTIONABLE feedback
- Feedback should be concise (1-3 sentences)
- Focus on the most critical issues first
- If loop count >= 3, return "complete"
</feedback_rules>
</tasks>

Output ONLY the JSON object, no additional text."""

    user_input = {
        "user_story_groups": state.get("user_story_groups", {}),
        "latest_dev_output": state.get("latest_dev_output", []),
        "rfd_output": state.get("rfd_output", {}),
        "latest_dev_loop_count": state.get("latest_dev_loop_count", 0),
        "rfd_loop_count": state.get("rfd_loop_count", 0)
    }
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(user_input, indent=2))
    ]
    
    model = init_chat_model(model="gpt-4.1")
    response = model.invoke(messages)
    
    try:
        feedback = parse_model_json(response.content)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        latest_dev_loop_count = state.get("latest_dev_loop_count", 0)
        rfd_loop_count = state.get("rfd_loop_count", 0)
        
        if latest_dev_loop_count >= 2 or rfd_loop_count >= 2:
            feedback = {
                "latest_dev_feedback": "complete",
                "rfd_feedback": "complete"
            }
        else:
            raise ValueError("validation_agent returned unparsable content") from exc
    
    if state.get("latest_dev_loop_count", 0) >= 3:
        feedback["latest_dev_feedback"] = "complete"
    if state.get("rfd_loop_count", 0) >= 3:
        feedback["rfd_feedback"] = "complete"
    
    return {
        "latest_dev_feedback": feedback.get("latest_dev_feedback", "complete"),
        "rfd_feedback": feedback.get("rfd_feedback", "complete")
    }


# ============================================================================
# SUMMARY HIGHLIGHTS AGENT
# ============================================================================

async def summary_highlights_agent(state: PreDiscussionState) -> PreDiscussionState:
    
    system_prompt = """<tasks>
<description>
You are the Validation and output Agent you are given:
Updated RFD List.
Updated Stories with latest_development field.
metadata of the board.
</description>
<objectives>
1. Cross scan the RFDs input and the user stories to make sure all the RFDs and stories are covered.
2. If everything make sense, if there are any discrepancies they should be fixed either on user stories or rfd list level
3. scan the snapshot (especially in the comments and worklog for each user story) for any new Risks , Follow-ups or Dependencies that are not present in the prior-day RFDs.
4. If the user story is moved to closed/completed or similar status, all related RFDs should be Closed.
5. If there is an RFD with a user story number that is not present in the user stories, it should be closed.
6. provide a new dictionary called Summary_highlights, this dictionary contains:
1. (sprint_goal_on_track : Yes/No if board type='scrum' or goal_on_track: Yes/No if  board type='kanban'  )
2. project_on_track: Yes/No
3. brief_summary: 2 to 3 lines of overall summary of the project including key outstanding issues, risks, dependencies, and any notable progress or concerns.
</objectives>
<outputs>
only output an updated list of RFDs like the input json, no extra text.
The only fields that should be updated on existing RFDs are: status, update, source, rule_id, close_date.
If a new RFD is identified, it should be added to the output list in the same format and all the fields must be populated except close_date.
rule_id should be null
Make sure all dates are following MM-DD-YYYY format.
Make sure story numbers are consistent between the RFDs and user stories.
Make sure source is reflecting the source of update, closure or creation of the RFD, the source can be multiple as well since its an array.
Make sure open_date is never empty.
Make sure risk/dependency/followup_detail is never null, it should have why is the item a risk/dependency/followup.
Make sure latest_development is present from output from previous agents.
The output must have all the input tags with their content even if you didn't change them <UserStories> [{'assignee': '', 'status': '','latest_development': '','Number':['','',...,'']}]
Adding 'TERMINATE' Keyword will help us in further processes
</outputs>
</tasks>

Output ONLY the JSON object with UserStories, RFDList, Summary_highlights"""

    user_input = {
        "user_story_groups": state.get("user_story_groups", {}),
        "latest_dev_output": state.get("latest_dev_output", []),
        "rfd_output": state.get("rfd_output", {})
    }
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(user_input, indent=2))
    ]
    
    model = init_chat_model(model="gpt-4.1")
    response = model.invoke(messages)
    
    try:
        output = parse_model_json(response.content)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError("summary_highlights_agent returned unparsable content") from exc
    
    if not isinstance(output, dict):
        output = {}
    
    user_stories = output.get("UserStories", state.get("latest_dev_output", []))
    rfd_list = output.get("RFDList", state.get("rfd_output", {}))
    summary_highlights = output.get("Summary_highlights", {})
    
    if not isinstance(summary_highlights, dict):
        summary_highlights = {}
    
    board_category = state.get("metadata", {}).get("board_category", "scrum")
    if board_category == "kanban":
        if "sprint_goal_on_track" in summary_highlights:
            summary_highlights["goal_on_track"] = summary_highlights.pop("sprint_goal_on_track")
        if "goal_on_track" not in summary_highlights:
            summary_highlights["goal_on_track"] = "Unknown"
    else:
        if "goal_on_track" in summary_highlights:
            summary_highlights["sprint_goal_on_track"] = summary_highlights.pop("goal_on_track")
        if "sprint_goal_on_track" not in summary_highlights:
            summary_highlights["sprint_goal_on_track"] = "Unknown"
    
    if "project_on_track" not in summary_highlights:
        summary_highlights["project_on_track"] = "Unknown"
    if "brief_summary" not in summary_highlights:
        summary_highlights["brief_summary"] = ""
    
    return {
        "UserStories": user_stories,
        "RFDList": rfd_list,
        "Summary_highlights": summary_highlights
    }


# ============================================================================
# FINALIZE NODE
# ============================================================================

def finalize_output_node(state: PreDiscussionState) -> PreDiscussionState:
    
    user_stories = state.get("UserStories", state.get("latest_dev_output", []))
    rfd_list = state.get("RFDList", state.get("rfd_output", {"risks": [], "followups": [], "dependencies": []}))
    summary_highlights = state.get("Summary_highlights", state.get("summary_highlights", {}))
    
    return {
        "UserStories": user_stories,
        "RFDList": rfd_list,
        "Summary_highlights": summary_highlights
    }


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def should_continue_validation(state: PreDiscussionState) -> str:
    
    latest_dev_feedback = state.get("latest_dev_feedback", "complete")
    rfd_feedback = state.get("rfd_feedback", "complete")
    latest_dev_loop_count = state.get("latest_dev_loop_count", 0)
    rfd_loop_count = state.get("rfd_loop_count", 0)
    
    if latest_dev_loop_count >= 3:
        latest_dev_feedback = "complete"
    
    if rfd_loop_count >= 3:
        rfd_feedback = "complete"
    
    if latest_dev_feedback == "complete" and rfd_feedback == "complete":
        return "summary"
    elif latest_dev_feedback != "complete" and rfd_feedback != "complete":
        return "retry_both"
    elif latest_dev_feedback != "complete":
        return "retry_latest_dev"
    else:
        return "retry_rfd"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

graph = StateGraph(state_schema=PreDiscussionState)

graph.add_node('preprocessing', preprocessing_node)
graph.add_node('latest_dev', latest_dev_agent)
graph.add_node('rfd', rfd_agent)
graph.add_node('validation', validation_agent)
graph.add_node('summary_highlights', summary_highlights_agent)
graph.add_node('finalize', finalize_output_node)

graph.add_edge(START, 'preprocessing')
graph.add_edge('preprocessing', 'latest_dev')
graph.add_edge('latest_dev', 'rfd')
graph.add_edge('rfd', 'validation')

graph.add_conditional_edges(
    'validation',
    should_continue_validation,
    {
        'summary': 'summary_highlights',
        'retry_latest_dev': 'latest_dev',
        'retry_rfd': 'rfd',
        'retry_both': 'latest_dev'
    }
)

graph.add_edge('summary_highlights', 'finalize')

graph.add_edge('finalize', END)

app = graph.compile()

__all__ = ['app']
