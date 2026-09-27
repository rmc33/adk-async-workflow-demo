# About
Run a workflow asynchronously from a chat agent. Demo using
ADK to start a workflow job. The
root agent can determine that the workflow should be run
based on the user's request. Allow the user to check
the status and retrieve results of the job.

I added an option to use Gemma3 local model for development. It can
cause unexpected behavior, for example calling tools without ending the turn, but still useful for development without incurring token costs. Switch to the Google Gemini model to test the expected behavior.

# Run with adk web
Recommended for development
```
adk web
```

# Run as cli app
```
python agent.py
```
