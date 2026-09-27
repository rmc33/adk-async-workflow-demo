import sys
import asyncio
import pathlib
import time
import uuid

from dotenv import load_dotenv
from google.adk import Agent, Context, Runner, Workflow
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.skills import load_skill_from_dir
from google.adk.tools import ToolContext
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.workflow import node
from google.genai import types
from pydantic import Field

from google.adk.models import Gemma3Ollama

load_dotenv()

_ENABLE_LOCAL_MODEL=False

if _ENABLE_LOCAL_MODEL:
    # ollama local Gemma 3 4b model for development
    _MODEL=Gemma3Ollama(model="ollama/gemma3:4b")
else:
    # cloud based Google Gemini 3.7 Flash model
    _MODEL="gemini-3.7-flash"


blog_writer_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "blog-writer"
)

content_researcher_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "content-research-writer"
)

blog_skills_agent = Agent(
    model=_MODEL,
    name="blog_skills_agent",
    description="A blog_skills_agent agent for writing blogs.",
    instruction=(
        "You are a blog-writing assistant with specialized skills.\n\n"
        "You have skills available:\n"
        "- **blog-writer**: Writing structure and style guide (load for writing)\n"
    ),
    tools=[SkillToolset(skills=[blog_writer_skill])],
)

content_researcher_agent = Agent(
    model=_MODEL,
    name="content_researcher_agent",
    description="A content_researcher_agent agent for researching blog topics.",
    instruction=(
        "You are a content researcher agent with specialized skills.\n\n"
        "You have skills available:\n"
        "- **content_researcher_agent**: Writing structure and style guide (load for writing)\n"
    ),
    tools=[SkillToolset(skills=[content_researcher_skill])],
)

@node(rerun_on_resume=True)
async def start_node(ctx: Context, node_input: str) -> str:
    return f"Starting workflow with job id {ctx.invocation_id} for input: {node_input}"

# Workflow chaining the start node, content researcher, and blog writer agents
job_workflow = Workflow(
    name="async_job_workflow",
    edges=[
        ("START", start_node, content_researcher_agent, blog_skills_agent)
    ],
)

_session_service: InMemorySessionService = None

def get_session_service() -> InMemorySessionService:
    global _session_service
    if _session_service is None:
        _session_service = InMemorySessionService()
    return _session_service

async def get_session(app_name: str, user_id: str, session_id: str) -> InMemorySessionService:
    session = await get_session_service().get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    if not session:
        # Create a session if not found. This can happen when running in adk web
        # when using In-Memory Session Service, since the adk web process will start the session
        # in it's own web server process. May stil be good to keep as a fail-safe
        # even when using a persistent session.
        print(f"Session not found for session {session_id}, creating session for user {user_id} for app {app_name}")
        session = await get_session_service().create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )
    return session

# Background runner worker
async def run_workflow_background(
    job_id: str,
    node_input: str,
    session_service: InMemorySessionService,
    session_id: str,
    app_name: str,
    user_id: str  # Ensure user_id matches caller
):
    """Executes the workflow out-of-band using the existing session."""

    # Yield control to allow the root agent turn to complete
    await asyncio.sleep(0)

    print(f"run_workflow_background for job {job_id} , app_name {app_name}, session_id {session_id}, user_id {user_id}, node_input {node_input}")
    runner = Runner(
        agent=job_workflow,
        app_name=app_name,
        session_service=session_service,
    )
    
    # Fetch existing session using the matching user_id
    session = await get_session(app_name, user_id, session_id)

    if not session:
        print(f"Background Job {job_id} error: Session '{session_id}' not found for user '{user_id}'.")
        return

    # Mark job as running in session state
    await session_service.append_event(session, Event(
        invocation_id=job_id, author="system", timestamp=time.time(),
        actions=EventActions(state_delta={f"job:{job_id}:status": "running"})
    ))

    try:
        result_texts = []
        async for event in runner.run_async(
            user_id=user_id,  # Use matching user_id
            session_id=session.id,
            invocation_id=job_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text=node_input)],
            ),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        result_texts.append(part.text)
        
        final_result = "\n".join(result_texts).strip()
        print(f"Job {job_id} completed with result: {final_result}")
        await session_service.append_event(session, Event(
            invocation_id=job_id, author="system", timestamp=time.time(),
            actions=EventActions(state_delta={
                f"job:{job_id}:status": "completed",
                f"job:{job_id}:result": final_result or "No output returned."
            })
        ))
    except Exception as e:
        await session_service.append_event(session, Event(
            invocation_id=job_id, author="system", timestamp=time.time(),
            actions=EventActions(state_delta={
                f"job:{job_id}:status": "failed",
                f"job:{job_id}:error": str(e)
            })
        ))
        print(f"Background Job {job_id} encountered an error: {e}")


async def kick_off_workflow_background_job(ctx: ToolContext, node_input: str) -> str:
    """Launches a background workflow bound to the current session."""
    job_id = f"job-{uuid.uuid4().hex[:8]}"

    session_id = ctx.session.id
    user_id = getattr(ctx.session, "user_id", "user")
    app_name = ctx.session.app_name
    session_service = get_session_service()

    # Schedule directly on the active event loop
    asyncio.create_task(
        run_workflow_background(
            job_id=job_id,
            node_input=node_input,
            app_name=app_name,
            session_service=session_service,
            session_id=session_id,
            user_id=user_id,
        )
    )

    session = await get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )

    await session_service.append_event(session, Event(
        invocation_id=job_id, author="system", timestamp=time.time(),
        actions=EventActions(state_delta={f"job:{job_id}:status": "running"})
    ))

    print(f"Starting job {job_id}, session id {session_id}, user id {user_id}, app_name {app_name}")    
    return f"async job started with id {job_id}"

# Tool to query job status and results
async def get_job_result(ctx: ToolContext, job_id: str) -> str:
    """Gets the status and result of a background job by querying session state directly."""
    session_service = get_session_service()
    session_id = ctx.session.id
    user_id = getattr(ctx.session, "user_id", "user")
    app_name = ctx.session.app_name

    print(f"Checking job {job_id}, session id {session_id}, user id {user_id}, app_name {app_name}")

    # Load fresh session state directly from the session service
    session = await get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    if not session or not session.state:
        return f"Job '{job_id}' not found."

    state = session.state
    status_key = f"job:{job_id}:status"

    if status_key not in state:
        return f"Job '{job_id}' not found."

    status = state.get(status_key)

    if status == "running":
        return f"Job '{job_id}' is still in progress."
    if status == "failed":
        error_msg = state.get(f"job:{job_id}:error", "Unknown error")
        return f"Job '{job_id}' failed with error: {error_msg}"

    result_data = state.get(f"job:{job_id}:result", "No result data available.")
    return f"Job '{job_id}' completed.\n\nResult:\n{result_data}"


# Root Agent configuration
root_agent = Agent(
    model=_MODEL,
    name="root_agent",
    description="A blog-writing assistant that runs research and writing workflows in the background.",
    instruction=(
        "You are a helpful blog-writing assistant.\n\n"
        "When a user asks to write or research a blog:\n"
        "1. Call `kick_off_workflow_background_job` ONCE with their request.\n"
        "2. IMMEDIATELY stop making tool calls and reply directly to the user with the exact job ID returned by the tool.\n"
        "3. DO NOT call `get_job_result` right after starting a job.\n\n"
        "When a user explicitly asks for the status or result of an existing job ID:\n"
        "1. Call `get_job_result` with that job_id.\n"
        "2. Return the job status to the user if the job is still pending and end the turn to wait for the next user request.\n"
        "3. Return the job result to the user if the job is completed or failed and end the turn to wait for the next user request."
    ),
    tools=[get_job_result, kick_off_workflow_background_job],
)

if __name__ == "__main__":
    async def chat_loop():
        session_service = get_session_service()
        runner = Runner(
            agent=root_agent,
            app_name="blog_app",
            session_service=session_service,
        )
        session = await session_service.create_session(
            app_name="blog_app",
            user_id="user",
        )
        print("=" * 60)
        print("🤖 Blog Agent Chat Started! (type 'exit' or 'quit' to end)")
        print("=" * 60)

        while True:
            try:
                user_prompt = input("\nYou > ").strip()
                if not user_prompt:
                    continue
                if user_prompt.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break

                print("\nAgent > ", end="", flush=True)
                async for event in runner.run_async(
                    user_id="user",
                    session_id=session.id,
                    new_message=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_prompt)],
                    ),
                ):
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                print(part.text, end="", flush=True)
                print()
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break

    asyncio.run(chat_loop())
