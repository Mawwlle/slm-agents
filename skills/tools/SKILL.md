---
name: function-calling
description: How to turn a user request into correct calls of the given functions.
---
1. Split the request into the separate actions it asks for. Each action is one call; "for A and B" or "then repeat with" means several calls.
2. Pick the function whose description fits each action. Use its name exactly as listed, dots included.
3. Fill every required argument. Take values from the request literally; convert them to the declared type (integer 5, not "5"; a list for an array). Do not invent optional arguments that the request does not mention.
4. Put all calls in one answer, one <tool_call> block per call.
5. If a call comes back with a problem (missing or wrong-typed argument), fix that call and send it again.
6. Then call finish.
